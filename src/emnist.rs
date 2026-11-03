// EMNIST ByClass Loader + Captcha-Stil-Daten
//
// Dataset: Royc30ne/emnist-byclass (HuggingFace)
//   62 Klassen: 0-9, A-Z, a-z
//   Train: 697.932   Test: 116.323
//
// Die vier EMNIST-.gz-Dateien einmal von Hand herunterladen und in einen Ordner legen:
//   https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main/emnist-byclass-train-images-idx3-ubyte.gz
//   https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main/emnist-byclass-train-labels-idx1-ubyte.gz
//   https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main/emnist-byclass-test-images-idx3-ubyte.gz
//   https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main/emnist-byclass-test-labels-idx1-ubyte.gz
//
// Captcha-Stil-Daten (gleiches Format, gleiche Klassen, ebenfalls transponiert)
// in denselben Ordner legen:
//   captcha-byclass-train-images-idx3-ubyte.gz
//   captcha-byclass-train-labels-idx1-ubyte.gz
//   captcha-byclass-test-images-idx3-ubyte.gz
//   captcha-byclass-test-labels-idx1-ubyte.gz
//
// Welche Daten geladen werden, bestimmt `Source`:
//   Source::Emnist    nur EMNIST (Verhalten wie bisher, `load()` macht genau das)
//   Source::Captcha   nur Captcha-Daten
//   Source::Combined  beide zusammen, gemischt
//
// Dependencies in Cargo.toml:
//   flate2 = "1"
//
// WICHTIG zur Speichernutzung: Pixel werden als u8 gehalten (1 Byte statt 4).
// EMNIST-Train sind damit ~547 MB statt ~2,2 GB. Die f32-Tensors entstehen erst
// pro Batch in batch_images(). Deswegen gibt es hier KEINEN Tensor mit allen
// Bildern - der würde den Speicher sprengen.

use std::io::Read;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

pub const DOWNLOAD_BASE: &str =
    "https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main";
pub const IMAGE_SIZE: usize = 28;
pub const PIXELS_PER_IMAGE: usize = IMAGE_SIZE * IMAGE_SIZE;

/// Reihenfolge laut emnist-byclass-mapping.txt:
/// Index 0-9 = Ziffern, 10-35 = Großbuchstaben, 36-61 = Kleinbuchstaben.
pub const VOCAB: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const MAGIC_IMAGES: u32 = 0x0000_0803;
const MAGIC_LABELS: u32 = 0x0000_0801;

/// Seed für das Mischen bei Source::Combined (fest -> reproduzierbar)
const SHUFFLE_SEED: u64 = 0x5EED_CAFE_1234_5678;

// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Emnist,
    Captcha,
    Combined,
}

struct DatasetFiles {
    label: &'static str,
    train_images: &'static str,
    train_labels: &'static str,
    test_images: &'static str,
    test_labels: &'static str,
    /// Download-Basis für die Fehlermeldung, None = lokal erzeugte Daten
    download_base: Option<&'static str>,
}

const EMNIST_FILES: DatasetFiles = DatasetFiles {
    label: "emnist",
    train_images: "emnist-byclass-train-images-idx3-ubyte.gz",
    train_labels: "emnist-byclass-train-labels-idx1-ubyte.gz",
    test_images: "emnist-byclass-test-images-idx3-ubyte.gz",
    test_labels: "emnist-byclass-test-labels-idx1-ubyte.gz",
    download_base: Some(DOWNLOAD_BASE),
};

const CAPTCHA_FILES: DatasetFiles = DatasetFiles {
    label: "captcha",
    train_images: "captcha-byclass-train-images-idx3-ubyte.gz",
    train_labels: "captcha-byclass-train-labels-idx1-ubyte.gz",
    test_images: "captcha-byclass-test-images-idx3-ubyte.gz",
    test_labels: "captcha-byclass-test-labels-idx1-ubyte.gz",
    download_base: None,
};

// ---------------------------------------------------------------------------

pub struct Split {
    /// count * 784 Pixel, zeilenweise pro Bild, Werte 0-255
    pub pixels: Vec<u8>,
    pub labels: Vec<u32>,
    pub count: usize,
}

pub struct Emnist {
    pub train: Split,
    pub test: Split,
}

// ---------------------------------------------------------------------------

impl Split {
    /// Baut den Bild-Tensor für einen Batch: (len, 1, 28, 28), f32 im Bereich 0.0..1.0
    pub fn batch_images(&self, start: usize, len: usize, device: &Device) -> Result<Tensor, String> {
        if start + len > self.count {
            return Err(format!(
                "batch {}..{} out of range, split has {} images",
                start,
                start + len,
                self.count
            ));
        }

        let from = start * PIXELS_PER_IMAGE;
        let to = (start + len) * PIXELS_PER_IMAGE;

        let values: Vec<f32> = self.pixels[from..to]
            .iter()
            .map(|&p| p as f32 / 255.0)
            .collect();

        Tensor::from_vec(values, (len, 1, IMAGE_SIZE, IMAGE_SIZE), device)
            .map_err(|e| e.to_string())
    }

    /// Baut den Label-Tensor für einen Batch: (len,), u32
    pub fn batch_labels(&self, start: usize, len: usize, device: &Device) -> Result<Tensor, String> {
        if start + len > self.count {
            return Err(format!(
                "batch {}..{} out of range, split has {} labels",
                start,
                start + len,
                self.count
            ));
        }

        let values: Vec<u32> = self.labels[start..start + len].to_vec();

        Tensor::from_vec(values, (len,), device).map_err(|e| e.to_string())
    }

    /// Ein Bild als ASCII auf die Konsole - zum Prüfen der Ausrichtung.
    pub fn preview(&self, index: usize) -> Result<(), String> {
        if index >= self.count {
            return Err(format!("index {} out of range ({} images)", index, self.count));
        }

        let chars: Vec<char> = VOCAB.chars().collect();
        let label = self.labels[index] as usize;
        let symbol = chars.get(label).copied().unwrap_or('?');

        println!("index {} | label {} = '{}'", index, label, symbol);

        let base = index * PIXELS_PER_IMAGE;
        let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '@'];

        for y in 0..IMAGE_SIZE {
            let mut line = String::with_capacity(IMAGE_SIZE);
            for x in 0..IMAGE_SIZE {
                let value = self.pixels[base + y * IMAGE_SIZE + x];
                line.push(ramp[(value as usize * 9 / 256).min(8)]);
            }
            println!("{}", line);
        }

        Ok(())
    }

    /// Hängt einen zweiten Split hinten an.
    fn append(&mut self, mut other: Split) {
        self.pixels.append(&mut other.pixels);
        self.labels.append(&mut other.labels);
        self.count += other.count;
    }

    /// Fisher-Yates: mischt Bilder und Labels gemeinsam.
    /// Nötig bei Combined, sonst kämen alle Captcha-Bilder am Ende und
    /// die letzten Batches jeder Epoche wären reine Captcha-Batches.
    fn shuffle(&mut self, seed: u64) {
        let mut rng = XorShift64::new(seed);

        for i in (1..self.count).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            if i != j {
                self.swap_images(i, j);
            }
        }
    }

    fn swap_images(&mut self, a: usize, b: usize) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (left, right) = self.pixels.split_at_mut(hi * PIXELS_PER_IMAGE);
        left[lo * PIXELS_PER_IMAGE..(lo + 1) * PIXELS_PER_IMAGE]
            .swap_with_slice(&mut right[..PIXELS_PER_IMAGE]);
        self.labels.swap(a, b);
    }

    fn truncate(&mut self, limit: usize) {
        if limit < self.count {
            self.count = limit;
            self.pixels.truncate(limit * PIXELS_PER_IMAGE);
            self.labels.truncate(limit);
        }
    }
}

/// Minimaler RNG, damit keine rand-Dependency nötig ist.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// ---------------------------------------------------------------------------

/// Lädt EMNIST ByClass aus einem Ordner, der die vier .gz-Dateien enthält.
/// Unverändertes Verhalten wie bisher, entspricht `load_from(dir, Source::Emnist, max_rows)`.
///
/// `max_rows`: optionale Obergrenze pro Split, um mit weniger Daten zu testen.
///             None = alles.
pub fn load(dir: &Path, max_rows: Option<usize>) -> Result<Emnist, String> {
    load_from(dir, Source::Emnist, max_rows)
}

/// Lädt EMNIST, Captcha-Daten oder beides gemischt aus `dir`.
///
/// `max_rows`: optionale Obergrenze pro Split (bei Combined nach dem Mischen,
///             d. h. die Teilmenge enthält beide Quellen anteilig).
pub fn load_from(dir: &Path, source: Source, max_rows: Option<usize>) -> Result<Emnist, String> {
    let (mut train, mut test) = match source {
        Source::Emnist => load_dataset(dir, &EMNIST_FILES)?,
        Source::Captcha => load_dataset(dir, &CAPTCHA_FILES)?,
        Source::Combined => {
            let (mut train, mut test) = load_dataset(dir, &EMNIST_FILES)?;
            let (captcha_train, captcha_test) = load_dataset(dir, &CAPTCHA_FILES)?;

            train.append(captcha_train);
            test.append(captcha_test);

            train.shuffle(SHUFFLE_SEED);
            test.shuffle(SHUFFLE_SEED ^ 0xFFFF);

            (train, test)
        }
    };

    if let Some(limit) = max_rows {
        train.truncate(limit);
        test.truncate(limit);
    }

    println!("{:?} -> train: {} Bilder, test: {} Bilder", source, train.count, test.count);

    Ok(Emnist { train, test })
}

fn load_dataset(dir: &Path, files: &DatasetFiles) -> Result<(Split, Split), String> {
    let train_images_path = require_file(dir, files.train_images, files.download_base)?;
    let train_labels_path = require_file(dir, files.train_labels, files.download_base)?;
    let test_images_path = require_file(dir, files.test_images, files.download_base)?;
    let test_labels_path = require_file(dir, files.test_labels, files.download_base)?;

    let train = load_split(
        &format!("{}/train", files.label),
        &train_images_path,
        &train_labels_path,
    )?;
    let test = load_split(
        &format!("{}/test", files.label),
        &test_images_path,
        &test_labels_path,
    )?;

    Ok((train, test))
}

fn require_file(dir: &Path, name: &str, download_base: Option<&str>) -> Result<PathBuf, String> {
    let path = dir.join(name);

    if !path.exists() {
        return Err(match download_base {
            Some(base) => format!("{} not found.\nDownload: {}/{}", path.display(), base, name),
            None => format!("{} not found.", path.display()),
        });
    }

    Ok(path)
}

fn load_split(name: &str, images_path: &Path, labels_path: &Path) -> Result<Split, String> {
    let image_bytes = read_gz(images_path)?;
    let label_bytes = read_gz(labels_path)?;

    let (mut pixels, image_count) = parse_idx3(&image_bytes)?;
    let (labels, label_count) = parse_idx1(&label_bytes)?;

    if image_count != label_count {
        return Err(format!(
            "{}: {} images but {} labels",
            name, image_count, label_count
        ));
    }

    // EMNIST (und die Captcha-Daten) liegen in den IDX-Dateien transponiert vor
    // (Zeilen und Spalten vertauscht gegenüber MNIST). Ohne diesen Schritt
    // stehen alle Zeichen seitwärts und das Netz lernt Unsinn.
    transpose_all(&mut pixels, image_count);

    println!("{}: {} Bilder", name, image_count);

    Ok(Split {
        pixels,
        labels,
        count: image_count,
    })
}

// ---------------------------------------------------------------------------

fn read_gz(path: &Path) -> Result<Vec<u8>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut buffer = Vec::new();

    decoder
        .read_to_end(&mut buffer)
        .map_err(|e| format!("{}: {}", path.display(), e))?;

    Ok(buffer)
}

fn be_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    if offset + 4 > bytes.len() {
        return Err(format!("unexpected end of file at offset {}", offset));
    }

    Ok(u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

/// IDX3: 4 Byte Magic, 4 Byte Anzahl, 4 Byte Höhe, 4 Byte Breite, dann die Pixel.
fn parse_idx3(bytes: &[u8]) -> Result<(Vec<u8>, usize), String> {
    let magic = be_u32(bytes, 0)?;
    if magic != MAGIC_IMAGES {
        return Err(format!("expected magic {:#010x}, got {:#010x}", MAGIC_IMAGES, magic));
    }

    let count = be_u32(bytes, 4)? as usize;
    let height = be_u32(bytes, 8)? as usize;
    let width = be_u32(bytes, 12)? as usize;

    if height != IMAGE_SIZE || width != IMAGE_SIZE {
        return Err(format!("expected 28x28 images, got {}x{}", height, width));
    }

    let expected = count * PIXELS_PER_IMAGE;
    let data = &bytes[16..];

    if data.len() < expected {
        return Err(format!(
            "file says {} images ({} bytes) but only {} bytes of data",
            count,
            expected,
            data.len()
        ));
    }

    Ok((data[..expected].to_vec(), count))
}

/// IDX1: 4 Byte Magic, 4 Byte Anzahl, dann ein Byte pro Label.
fn parse_idx1(bytes: &[u8]) -> Result<(Vec<u32>, usize), String> {
    let magic = be_u32(bytes, 0)?;
    if magic != MAGIC_LABELS {
        return Err(format!("expected magic {:#010x}, got {:#010x}", MAGIC_LABELS, magic));
    }

    let count = be_u32(bytes, 4)? as usize;
    let data = &bytes[8..];

    if data.len() < count {
        return Err(format!(
            "file says {} labels but only {} bytes of data",
            count,
            data.len()
        ));
    }

    let vocab_len = VOCAB.chars().count() as u32;
    let mut labels = Vec::with_capacity(count);

    for &raw in &data[..count] {
        let label = raw as u32;
        if label >= vocab_len {
            return Err(format!("label {} outside vocab of {} classes", label, vocab_len));
        }
        labels.push(label);
    }

    Ok((labels, count))
}

/// Vertauscht in jedem Bild Zeilen und Spalten: out[y][x] = in[x][y]
fn transpose_all(pixels: &mut [u8], count: usize) {
    let mut scratch = vec![0u8; PIXELS_PER_IMAGE];

    for index in 0..count {
        let base = index * PIXELS_PER_IMAGE;
        let image = &mut pixels[base..base + PIXELS_PER_IMAGE];

        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                scratch[y * IMAGE_SIZE + x] = image[x * IMAGE_SIZE + y];
            }
        }

        image.copy_from_slice(&scratch);
    }
}