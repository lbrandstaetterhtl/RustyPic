use std::collections::HashMap;
use std::path::Path;
use candle_core::{DType, Device, Tensor};
use candle_nn::Optimizer;
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use crate::models::{AppConfig, BBox, CNN};
use crate::emnist;
use crate::emnist::Emnist;
use imageproc::region_labelling::{connected_components, Connectivity};

pub fn grayscale(source_path: String, output_path: String) -> Result<Option<String>, String>
{
    let source_result = image::open(source_path);
    let source = match source_result {
        Ok(source) => source,
        Err(error) => return Err(error.to_string())
    };
    
    let gray = source.grayscale();
    
    let gray_result = gray.save(&output_path);
    
    match gray_result {
        Ok(_) => Ok(Some(format!("Saved gray scaled image to {}", output_path))),
        Err(error) => Err(error.to_string())
    }
}

pub fn invert(source_path: String, output_path: String) -> Result<Option<String>, String> {
    let source_result = image::open(source_path);

    let mut source = match source_result {
        Ok(source) => source,
        Err(error) => return Err(error.to_string())
    };

   source.invert();

    let save = source.save(&output_path);

    match save {
        Ok(_) => Ok(Some(format!("Saved inverted image to {}", output_path))),
        Err(error) => Err(error.to_string())
    }
}

pub fn test_ocr() -> Result<Option<String>, String> {
    let app_config = AppConfig::load().map_err(|error| "[app_config]".to_string() + error.as_str())?;
    let device = app_config.device()?;
    let data = load_data().map_err(|error| "[data]".to_string() + error.as_str())?;
    let cnn = CNN::load(&device).map_err(|error| "[cnn]".to_string() + error.as_str())?;

    let bsz = app_config.batch_size;
    let n_batches = data.test.count / bsz;

    let mut correct = 0usize;
    let mut seen = 0usize;
    let mut confusion: HashMap<(u32, u32), usize> = HashMap::new();

    for batch in 0..n_batches {
        let start = batch * bsz;
        let images = data.test.batch_images(start, bsz, &device)?;
        let labels = data.test.batch_labels(start, bsz, &device)?;

        let logits = cnn.forward(images)?;
        let pred = logits.argmax(candle_core::D::Minus1).map_err(|e| e.to_string())?
            .to_vec1::<u32>().map_err(|e| e.to_string())?;
        let truth = labels.to_vec1::<u32>().map_err(|e| e.to_string())?;

        for i in 0..pred.len() {
            if pred[i] == truth[i] {
                correct += 1;
            } else {
                *confusion.entry((truth[i], pred[i])).or_insert(0) += 1;
            }
        }
        seen += pred.len();

        if batch % 200 == 0 {
            print!("  batch {batch:4} | predicted: ");
            for i in 0..10 {
                print!("{}/{} ", pred[i], cnn.vocab.decode(pred[i]));
            }
            print!("| expected: ");
            for i in 0..10 {
                print!("{}/{} ", truth[i], cnn.vocab.decode(truth[i]));
            }
            print!("\n");
        }
    }

    println!("test accuracy: {:.2}%  ({} von {})",
             correct as f32 / seen as f32 * 100.0, correct, seen);

    let chars: Vec<char> = emnist::VOCAB.chars().collect();
    let mut pairs: Vec<_> = confusion.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nMost fails:");
    for ((truth, pred), count) in pairs.iter().take(15) {
        println!("  '{}' als '{}'  {}x",
                 chars[*truth as usize], chars[*pred as usize], count);
    }

    Ok(None)
}

pub fn resolve(source_path: String) -> Result<Option<String>, String> {
    let app_config = match AppConfig::load() {
        Ok(app_config) => app_config,
        Err(error) => return Err(error.to_string())
    };
    let device = match app_config.device() {
        Ok(device) => device,
        Err(error) => return Err(error.to_string())
    };
    let cnn = match CNN::load(&device) {
        Ok(cnn) => cnn,
        Err(error) => return Err(error.to_string())
    };

    let tensors = match load_and_split(source_path.as_str(), device) {
        Ok(tensors) => tensors,
        Err(error) => return Err(error.to_string())
    };


    let mut full_prediction: String = String::new();

    for tensor in tensors.iter() {
        let pred = tensor.argmax(candle_core::D::Minus1).map_err(|error| error.to_string())?.to_vec1::<u32>().map_err(|error| error.to_string())?[0];
        let c = cnn.vocab.decode(pred);
        full_prediction.push(c);
    }

    Ok(Some(full_prediction))
}

pub fn train_ocr() -> Result<Option<String>, String> {

    let data = match load_data() {
        Ok(data) => data,
        Err(error) => return Err(error.to_string())
    };

    let app_config = match AppConfig::load() {
        Ok(app_config) => app_config,
        Err(error) => return Err(error.to_string())
    };

    let device = match app_config.device() {
        Ok(device) => device,
        Err(error) => return Err(error.to_string())
    };

    let cnn = match CNN::load(&device) {
        Ok(cnn) => cnn,
        Err(error) => return Err(error.to_string())
    };

    let mut optimizer = match candle_nn::AdamW::new_lr(cnn.var_map.all_vars(), app_config.learning_rate) {
        Ok(optimizer) => optimizer,
        Err(error) => return Err(error.to_string())
    };

    let n_batches = data.train.count / app_config.batch_size;

    let mut lr = app_config.learning_rate;
    for epoch in 1..=app_config.epochs {
        let mut sum_loss = 0f32;
        let mut correct = 0usize;
        let mut seen = 0usize;

        for batch in 0..n_batches {
            let start = batch * app_config.batch_size;

            let images = data.train.batch_images(start, app_config.batch_size, &device)?;
            let labels = data.train.batch_labels(start, app_config.batch_size, &device)?;

            let logits = cnn.forward(images).map_err(|error| error.to_string())?;
            let loss = candle_nn::loss::cross_entropy(&logits, &labels).map_err(|error| error.to_string())?;
            optimizer.backward_step(&loss).map_err(|error| error.to_string())?;

            sum_loss += loss.to_vec0::<f32>().unwrap();
            let pred = logits.argmax(candle_core::D::Minus1).map_err(|error| error.to_string())?.to_vec1::<u32>().map_err(|error| error.to_string())?;
            let truth = labels.to_vec1::<u32>().map_err(|error| error.to_string())?;

            for i in 0..pred.len() {
                if pred[i] == truth[i] {
                    correct += 1;
                }
            }
            seen += pred.len();

            if batch % 200 == 0 {
                print!("  batch {batch:4} | predicted: ");
                for i in 0..10 {
                    print!("{}/{} ", pred[i], cnn.vocab.decode(pred[i]));
                }
                print!("| expected: ");
                for i in 0..10 {
                    print!("{}/{} ", truth[i], cnn.vocab.decode(truth[i]));
                }
                println!("| loss {:.4}", loss.to_vec0::<f32>().unwrap());
            }

        }
        println!(
            "epoch {epoch}| lr {lr:.5} | loss {:.4} | train accuracy {:.2}%",
            sum_loss / n_batches as f32,
            correct as f32 / seen as f32 * 100.0
        );

        lr = app_config.learning_rate * 0.5f64.powi((epoch as i32 - 1) / 3);
        optimizer.set_learning_rate(lr);
    }

    let _ = match cnn.save() {
        Ok(result) => result,
        Err(e) => return Err(e)
    };

    Ok(Some("Training finished and model saved successfully".to_string()))
}

fn load_and_split(path: &str, device: Device) -> Result<Vec<Tensor>, String> {
    let img = load_image(path.to_string()).map_err(|error| error.to_string())?;

    let letters = image_clean_and_split_up(img);

    let mut tensors: Vec<Tensor> = Vec::new();
    //DEBUG save
    let counter = 0;
    for letter in letters {
        letter.save(path.to_string() + "_" + counter.to_string().as_str()).map_err(|error| error.to_string())?;

        let pixel: Vec<f32> = letter.into_raw().iter()
            .map(|&p| p as f32 / 255.0).collect();

        let t = Tensor::from_vec(pixel, (1, 1, 28, 28), &device).map_err(|error| error.to_string())?;

        tensors.push(t);
    }

    Ok(tensors)
}

fn load_image(path: String) -> Result<GrayImage, String> {
    let img = image::open(path)
        .map_err(|e| e.to_string())?
        .to_luma8();
    Ok(img)
}

fn load_data() -> Result<Emnist, String> {
    let config_dir = match AppConfig::get_config_dir() {
        Ok(config_dir) => config_dir,
        Err(error) => return Err(error.to_string())
    };
    let emnist_dir = Path::new(&config_dir).join("emnist");

    let data = match emnist::load(&emnist_dir, None) {
        Ok(data) => data,
        Err(error) => return Err(error.to_string())
    };

    Ok(data)
}


fn image_clean_and_split_up(image: GrayImage) -> Vec<GrayImage> {
    let mut black_and_white = image;

    let threshold: f32 = 0.65;

    for pixel in black_and_white.pixels_mut() {
        let luma = pixel.0[0];

        let brightness = luma as f32 / 255.0;

        if brightness > threshold {
            *pixel = Luma([0]);
        }
        else {
            *pixel = Luma([255]);
        }
    }

    let letters = image_split_up(black_and_white);

    letters
}

fn image_split_up(image: GrayImage) -> Vec<GrayImage> {
    let labels = connected_components(&image, Connectivity::Eight, Luma([255])).to_vec();

    let min_pixels = 20;
    let bboxes: Vec<BBox> = get_bounding_boxes(&[labels])
        .into_iter()
        .filter(|b| b.width() * b.height() >= min_pixels)
        .collect();

    let mut result: Vec<GrayImage> = Vec::new();

    for bbox in bboxes {
        result.push(crop_letter(&image, &bbox));
    }

    result
}

fn get_bounding_boxes(labels: &[Vec<u32>]) -> Vec<BBox> {
    let mut boxes: HashMap<u32, BBox> = HashMap::new();

    for (y, row) in labels.iter().enumerate() {
        for (x, &label) in row.iter().enumerate() {
            if label == 0 {continue;}

            let x = x as u32;
            let y = y as u32;

            boxes.entry(label).and_modify(|b| {
                b.min_x = b.min_x.min(x);
                b.max_x = b.max_x.max(x);
                b.min_y = b.min_y.min(y);
                b.max_y = b.max_y.max(y);
            }).or_insert(BBox {min_x: x, min_y: y, max_x: x, max_y: y});
        }
    }

    let mut result: Vec<BBox> = boxes.into_values().collect();
    result.sort_by_key(|b| b.min_x);
    result
}
fn crop_letter(src: &GrayImage, b: &BBox) -> GrayImage {
    src.view(b.min_x, b.min_y, b.width(), b.height()).to_image()
}
