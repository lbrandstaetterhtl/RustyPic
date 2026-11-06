use std::collections::HashMap;
use std::path::Path;
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_nn::Optimizer;
use image::{imageops, DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::filter::median_filter;
use crate::models::{AppConfig, BBox, OcrTrainTestEnum, CNN};
use crate::emnist;
use crate::emnist::{Emnist, Source, load_from};
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

pub fn test_ocr(option: OcrTrainTestEnum) -> Result<Option<String>, String> {
    let app_config = AppConfig::load().map_err(|error| "[app_config]".to_string() + error.as_str())?;
    let device = app_config.device()?;
    let data = load_data(option).map_err(|error| "[data]".to_string() + error.as_str())?;
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

        let logits = cnn.forward(&images)?;
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

    let mut per_position: Vec<Vec<(char, f32)>> = Vec::new();
    for tensor in tensors.iter() {
        let logits = cnn.forward(tensor).map_err(|error| error.to_string())?;
        per_position.push(top_k(&logits, 3, &cnn)?)
    }

    let ranked = candidates(&per_position, 10);
    for (s, p) in ranked {
        full_prediction.push_str(format!("\n{}: {}", p, s).as_str());
    }

    Ok(Some(full_prediction))
}

fn top_k(logits: &Tensor, k: usize, cnn: &CNN) -> Result<Vec<(char, f32)>, String> {
    let probs: Vec<f32> = softmax(&logits.squeeze(0).map_err(|e| e.to_string())?, candle_core::D::Minus1)
        .map_err(|e| e.to_string())?
        .to_vec1()
        .map_err(|e| e.to_string())?;

    let vocab: Vec<char> = cnn.vocab.to_char.clone();
    let mut ranked: Vec<(char, f32)> = probs.into_iter().enumerate().map(|(i, p)| (vocab[i], p)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);
    Ok(ranked)
}

fn candidates(per_position: &[Vec<(char, f32)>], max: usize) -> Vec<(String, f32)> {
    let mut result: Vec<(String, f32)> = vec![(String::new(), 0.0)]; // Log-Wahrscheinlichkeit

    for options in per_position {
        let mut next = Vec::with_capacity(result.len() * options.len());
        for (prefix, logp) in &result {
            for &(ch, p) in options {
                let mut s = prefix.clone();
                s.push(ch);
                next.push((s, logp + p.max(1e-9).ln()));
            }
        }
        next.sort_by(|a, b| b.1.total_cmp(&a.1));
        next.truncate(max); // Beam: nur die besten behalten
        result = next;
    }

    result.into_iter().map(|(s, logp)| (s, logp.exp())).collect()
}

fn to_emnist(crop: &GrayImage) -> GrayImage {
    let (cw, ch) = crop.dimensions();
    let scale = 20.0 / cw.max(ch) as f32;
    let nw = ((cw as f32 * scale).round() as u32).clamp(1, 20);
    let nh = ((ch as f32 * scale).round() as u32).clamp(1, 20);
    let resized = imageops::resize(crop, nw, nh, imageops::FilterType::Triangle);

    let mut out = GrayImage::from_pixel(28, 28, Luma([0]));
    let dx = (28 - nw) / 2;
    let dy = (28 - nh) / 2;
    imageops::replace(&mut out, &resized, dx as i64, dy as i64);
    out
}

pub fn train_ocr(option: OcrTrainTestEnum) -> Result<Option<String>, String> {

    let data = match load_data(option) {
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

            let logits = cnn.forward(&images).map_err(|error| error.to_string())?;
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

pub fn load_and_split(path: &str, device: Device) -> Result<Vec<Tensor>, String> {
    let img = load_image(path.to_string()).map_err(|error| error.to_string())?;


    let letters = image_clean_and_split_up(img).map_err(|error| error.to_string())?;

    let mut tensors: Vec<Tensor> = Vec::new();
    for letter in letters {
        let pixel: Vec<f32> = letter.into_raw().iter()
            .map(|&p| p as f32 / 255.0).collect();

        let t = Tensor::from_vec(pixel, (1, 1, 28, 28), &device).map_err(|error| error.to_string())?;

        tensors.push(t);
    }

    Ok(tensors)
}

fn load_image(path: String) -> Result<DynamicImage, String> {
    let img = image::open(path)
        .map_err(|e| e.to_string())?;
    Ok(img)
}

fn load_data(option: OcrTrainTestEnum) -> Result<Emnist, String> {
    let config_dir = match AppConfig::get_config_dir() {
        Ok(config_dir) => config_dir,
        Err(error) => return Err(error.to_string())
    };
    let emnist_dir = Path::new(&config_dir).join("emnist");

    let data = match emnist::load_from(&emnist_dir, option.to_data_source().map_err(|e| e)?, None) {
        Ok(data) => data,
        Err(error) => return Err(error.to_string())
    };

    Ok(data)
}


fn image_clean_and_split_up(image: DynamicImage) -> Result<Vec<GrayImage>, String> {

    let gray = image.grayscale();
    let mut black_and_white = make_black_and_white(&gray.to_rgb8());
    black_and_white = median_filter(&black_and_white, 1, 1);
    black_and_white = imageproc::morphology::open(&black_and_white, imageproc::distance_transform::Norm::LInf, 1);

    let letters = image_split_up(black_and_white).map_err(|error| error.to_string())?;

    Ok(letters)
}

fn make_black_and_white(image: &RgbImage) -> GrayImage {
    let bg = background_color(&image);

    let max_dist = (3.0f32 * 255.0 * 255.0).sqrt();
    let (w, h) = image.dimensions();
    let mut black_and_white = GrayImage::new(w, h);
    for (x, y, px) in image.enumerate_pixels() {
        let Rgb([r, g, b]) = *px;
        let dr = r as f32 - bg[0];
        let dg = g as f32 - bg[1];
        let db = b as f32 - bg[2];
        let d = (dr * dr + dg * dg + db * db).sqrt() / max_dist * 255.0;
        black_and_white.put_pixel(x, y, Luma([if d > 32.0 { 255 } else { 0 }]));
    }

    black_and_white
}

fn background_color(img: &RgbImage) -> [f32; 3] {
    let mut ch: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for p in img.pixels() {
        for c in 0..3 {
            ch[c].push(p.0[c]);
        }
    }
    let mut bg = [0.0f32; 3];
    for c in 0..3 {
        ch[c].sort_unstable();
        bg[c] = ch[c][ch[c].len() / 2] as f32;
    }
    bg
}

fn image_split_up(image: GrayImage) -> Result<Vec<GrayImage>, String> {
    let labels = connected_components(&image.clone(), Connectivity::Eight, Luma([0]));

    let min_area = 30;
    let min_fill = 0.15;
    let mut bboxes: Vec<BBox> = get_bounding_boxes(&labels)
        .into_iter()
        .filter(|b| b.area >= min_area)
        .filter(|b| b.fill() >= min_fill)
        .collect();

    let max_h = bboxes.iter().map(|b| b.height()).max().unwrap_or(0);
    bboxes.retain(|b| b.height() * 10 >= max_h * 4);

    let mut result: Vec<GrayImage> = Vec::new();

    for bbox in bboxes {
        result.push(match crop_letter(&image.clone(), &bbox, 3).map(|c| to_emnist(&c)) {
            Some(result) => result,
            None => continue,
        });
    }

    Ok(result)
}

fn get_bounding_boxes(labels: &ImageBuffer<Luma<u32>, Vec<u32>>) -> Vec<BBox> {
    let mut boxes: HashMap<u32, BBox> = HashMap::new();

    for (x, y, px) in labels.enumerate_pixels() {
        let label = px.0[0];
        if label == 0 {
            continue;
        }

        boxes
            .entry(label)
            .and_modify(|b| {
                b.min_x = b.min_x.min(x);
                b.max_x = b.max_x.max(x);
                b.min_y = b.min_y.min(y);
                b.max_y = b.max_y.max(y);
                b.area += 1;
            })
            .or_insert(BBox { min_x: x, min_y: y, max_x: x, max_y: y, area: 1});
    }

    let mut result: Vec<BBox> = boxes.into_values().collect();
    result.sort_by_key(|b| b.min_x);
    result
}
fn crop_letter(src: &GrayImage, b: &BBox, pad: u32) -> Option<GrayImage> {
    let (w, h) = src.dimensions();

    let left = b.min_x.saturating_sub(pad);
    let top = b.min_y.saturating_sub(pad);
    let right = (b.max_x + 1 + pad).min(w);
    let bottom = (b.max_y + 1 + pad).min(h);

    if right <= left || bottom <= top {
        return None;
    }

    Some(imageops::crop_imm(src, left, top, right - left, bottom - top).to_image())
}