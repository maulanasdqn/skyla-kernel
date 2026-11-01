use std::collections::HashMap;

pub fn byte_chars() -> [char; 256] {
    let printable = |b: u32| (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
    let mut out = ['\0'; 256];
    let mut extra = 0;
    for b in 0..256u32 {
        out[b as usize] = match printable(b) {
            true => char::from_u32(b).unwrap(),
            false => {
                extra += 1;
                char::from_u32(255 + extra).unwrap()
            }
        };
    }
    out
}

pub fn merge(word: &str, ranks: &HashMap<(String, String), u32>) -> Vec<String> {
    let mut symbols: Vec<String> = word.chars().map(String::from).collect();
    loop {
        let best = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| ranks.get(&(pair[0].clone(), pair[1].clone())).map(|&rank| (rank, i)))
            .min();
        let Some((_, i)) = best else {
            return symbols;
        };
        let right = symbols.remove(i + 1);
        symbols[i].push_str(&right);
    }
}
