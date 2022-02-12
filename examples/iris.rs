use ami::{activation::Sigmoid, layer::Linear, network::Network, util::train_test_split};
use csv::Reader;
use ndarray::{s, stack, Array2, ArrayView2, Axis};
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
    let xs = Array2::from_shape_vec((n_records, 4), xs).unwrap();
    let ys = Array2::from_shape_vec((n_records, 3), ys).unwrap();
    (xs, ys)
}

fn accuracy(y_test: ArrayView2<f64>, y_pred: ArrayView2<f64>) -> f64 {
    assert_eq!(y_test.len_of(Axis(0)), y_pred.len_of(Axis(0)));
    let errors = &y_test - &y_pred;
    let mut n_corrects = 0;
    for error in errors.outer_iter() {
        if error.iter().all(|y| y.abs() < 0.1) {
            n_corrects += 1;
        }
    }
    (n_corrects as f64) / (y_test.len_of(Axis(0)) as f64)
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let row_sum = x.map_axis(Axis(1), |row| row.map(|v| v.exp()).iter().sum());
    dbg!(row_sum.shape());
    for i in 0..x.nrows() {
        //         *x.slice_mut(s![i, ..]) /= row_sum.get(i).unwrap();
    }
    x / row_sum
}

fn main() {
    let (xs, ys) = load_iris("./IRIS.csv");
    let (x_train, x_test, y_train, y_test) = train_test_split(xs, ys, 0.8);

    let batch_size = 12;
    let mut network = Network::new(0.5);
    network.add_layer(Linear::new(4, 5, batch_size, Sigmoid));
    network.add_layer(Linear::new(5, 3, batch_size, Sigmoid));

    let epochs = 10;
    for epoch in 0..epochs {
        let mut has_processed = 0;
        while has_processed < x_train.len_of(Axis(0)) {
            print!("#");
            let x_train_batch = x_train
                .slice(s![has_processed..(has_processed + batch_size), ..])
                .to_owned();
            let y_train_batch = y_train
                .slice(s![has_processed..(has_processed + batch_size), ..])
                .to_owned();
            let y_hat = network.forward(x_train_batch);
            let y_hat = softmax(&y_hat);
            let error = y_train_batch - y_hat;
            network.backward(error);
            has_processed += batch_size;
        }
        let y_pred = network.forward(x_train.clone());
        let y_pred = softmax(&y_pred);
        let acc = accuracy(y_train.view(), y_pred.view());
        println!("train acc: {}", acc);
        println!("{} epoch done", epoch);
    }

    let y_pred = network.forward(x_test);
    let acc = accuracy(y_test.view(), y_pred.view());
    println!("test acc: {}", acc);
    dbg!(y_pred, y_test);
}
