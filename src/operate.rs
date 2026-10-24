pub fn grayscale(source_path: String, output_path: String) -> Result<Option<String>, String>
{
    let source_result = image::open(source_path);
    let source = match source_result {
        Ok(source) => source,
        Err(error) => return Err(error.to_string())
    };
    
    let gray = source.grayscale();
    
    let gray_result = gray.save(output_path.clone());
    
    match gray_result {
        Ok(_) => Ok(Some(format!("Saved gray scaled image to {}", output_path))),
        Err(error) => Err(error.to_string())
    }
}