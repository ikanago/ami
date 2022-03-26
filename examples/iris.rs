use ami::{
    data::DataLoader,
    grad::{softmax_cross_entropy, Function, Variable},
    metrics::{accuracy, confusion_matrix},
    model::{Chainable, Input, Linear, Model, Relu},
    optimizer::GradientDescent,
    sequential,
    utils::train_test_split,
    OneHotEncoder,
};
use csv::Reader;
use ndarray::Array2;
use std::path::Path;

// Load iris dataset available here: https://www.kaggle.com/arshid/iris-flower-dataset
fn load_iris(file_path: impl AsRef<Path>) -> (Vec<Vec<f32>>, Vec<String>) {
    let mut reader = Reader::from_path(file_path).unwrap();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for row in reader.records() {
        let row = row.unwrap();
        let y = row[4].to_string();
        let features = row
            .into_iter()
            .take(4)
            .map(|r| r.parse().unwrap())
            .collect();
        xs.push(features);
        ys.push(y);
    }
    (xs, ys)
}

fn main() {
    let (xs, ys) = load_iris("./IRIS.csv");
    let labels = vec!["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let encoder = OneHotEncoder::new(&labels);

    let (x_train, y_train, x_test, y_test) = train_test_split(xs, ys, 0.25);
    let x_train =
        Array2::from_shape_vec((x_train.len(), 4), x_train.into_iter().flatten().collect())
            .unwrap();
    let y_train = encoder.encode(&y_train);
    let x_test =
        Array2::from_shape_vec((x_test.len(), 4), x_test.into_iter().flatten().collect()).unwrap();

    let mut loader = DataLoader::new(x_train, y_train).shuffle();

    let model = sequential!(
        Input,
        Linear::new(4, 10),
        Relu::new(),
        Linear::new(10, 10),
        Relu::new(),
        Linear::new(10, 3)
    );
    let optimizer = GradientDescent::new(1e-2);

    let epochs = 3000;
    let batch_size = 16;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (input, target) in loader.batch(batch_size) {
            let input = Variable::new(input);
            let target = Variable::new(target);

            let network = model.forward(input);
            let loss = softmax_cross_entropy(&network, &target);
            loss.forward();
            loss.backward();
            model.update_parameters(&optimizer);
            model.zero_gradient();

            total_loss += loss.data()[()];
        }

        if epoch % 10 == 0 {
            println!("epoch {}: total loss = {}", epoch, total_loss);
        }
    }

    let y_pred = model.forward(Variable::new(x_test));
    y_pred.forward();
    let y_pred = encoder.decode(y_pred.data().clone());
    let confusion_matrix = confusion_matrix(&y_test, &y_pred, &labels);

    println!("accuracy: {}", accuracy(&y_test, &y_pred));
    println!("{:?}", confusion_matrix);
}
