use std::fs::File;
use std::io::prelude::*;
use std::collections::{HashSet, HashMap};
use rand::{thread_rng, Rng};
use candle_core::{Device, Result, Tensor};

pub struct BigramLanguageModel {
}

impl BigramLanguageModel {
    pub fn new() -> Self {
        Self { }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.to_owned())
    }    
}

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

fn get_batch(dataset: &Vec<usize>) -> (Vec<&[usize]>, Vec<&[usize]>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let batch_size = 8;

    let rand_num: Vec<usize> = thread_rng()
        .sample_iter(&rand::distributions::Uniform::new(0, dataset.len()))
        .take(4)
        .collect();    

    for i in rand_num.into_iter() {
        xs.push(&dataset[i..i+batch_size]);
        ys.push(&dataset[i+1..i+batch_size+1]);
    }

    (xs, ys)
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
    let _test = encoded[train_size..].to_owned();

    let block_size = 8;

    println!("train: {:?}", decode(&train[0..block_size+1], &idx_to_char));

    let (xs, ys) = get_batch(&train);

    println!("X: {:?}", xs);
    println!("y: {:?}", ys);

    let llm = BigramLanguageModel::new();

    let data: [u32; 3] = [1u32, 2, 3];
    let input_tensor = Tensor::new(&data, &Device::Cpu).unwrap();
    let activation_tensor = llm.forward(&input_tensor).unwrap();

    println!("Activation: {:?}", activation_tensor);

    Ok(())
}
