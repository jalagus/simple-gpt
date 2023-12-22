use std::fs::File;
use std::io::prelude::*;
use std::collections::{HashSet, HashMap};

fn encode(text: &str, char_to_idx: &HashMap<char, usize>) -> Vec<usize> {
    let mut encoded = Vec::new();
    for c in text.chars() {
        encoded.push(*char_to_idx.get(&c).unwrap());
    }
    encoded
}

fn decode(encoded: &[usize], idx_to_char: &HashMap<usize, char>) -> String {
    let mut decoded = String::new();
    for i in encoded {
        decoded.push(*idx_to_char.get(i).unwrap());
    }
    decoded
}

fn main() -> std::io::Result<()> {
    let mut chars = HashSet::new();
    let mut char_to_idx = HashMap::new();
    let mut idx_to_char = HashMap::new();

    let mut file = File::open("input.txt")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    for c in contents.chars() {
        chars.insert(c);
    }

    for (i, c) in chars.iter().enumerate() {
        char_to_idx.insert(c.to_owned(), i);
        idx_to_char.insert(i, c.to_owned());
    }

    let _vocab_size = chars.len();

    let encoded = encode(&contents, &char_to_idx);

    let train_size = (encoded.len() as f32 * 0.9) as usize;
    let train = encoded[0..train_size].to_owned();
    let test = encoded[train_size..].to_owned();

    let block_size = 8;

    println!("train: {:?}", decode(&train[0..block_size+1], &idx_to_char));

    Ok(())
}
