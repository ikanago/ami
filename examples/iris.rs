use ami::network::Network;
use csv::Reader;
use ndarray::Array2;
use std::path::Path;

fn convert_iris_kinds_to_one_hot(kind: &str) -> Vec<f64> {
    match kind {
        "Iris-setosa" => vec![1.0, 0.0, 0.0],
        "Iris-versicolor" => vec![0.0, 1.0, 0.0],
        "Iris-virginica" => vec![0.0, 0.0, 1.0],
        _ => panic!("Unknown iris kind: {}", kind),
    }
}

// Load iris dataset available here: https://www.kaggle.com/arshid/iris-flower-dataset
fn load_iris(file_path: impl AsRef<Path>) -> (Array2<f64>, Array2<f64>) {
    let mut reader = Reader::from_path(file_path).unwrap();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut n_records = 0;
    for row in reader.records() {
        let row = row.unwrap();
        n_records += 1;
        for i in 0..4 {
            xs.push(row[i].parse::<f64>().unwrap());
        }
        let y = convert_iris_kinds_to_one_hot(&row[4]);
        ys.extend(y.iter());
    }
    dbg!(ys.len(), n_records);
    let xs = Array2::from_shape_vec((n_records, 4), xs).unwrap();
    let ys = Array2::from_shape_vec((n_records, 3), ys).unwrap();
    (xs, ys)
}

fn main() {
    let mut network = Network::new(0.5);
    let (xs, ys) = load_iris("./IRIS.csv");
    dbg!(xs);
    dbg!(ys);
}
