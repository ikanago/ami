use ami::{
    activation::{Identity, Relu},
    layer::Linear,
    network::Network,
};
use ndarray::{s, Array2, ArrayView2, Axis};
use ndarray_rand::{
    rand::{thread_rng, Rng},
    rand_distr::{Distribution, Normal, Uniform},
};

// x: (batch_size, input_size)
fn func(x: ArrayView2<f64>) -> Array2<f64> {
    x.map_axis(Axis(1), |v| {
        2.0 + 5.0 * v[[0]] + -3.0 * v[[1]].sin() + 4.0 * v[[2]].powi(2)
    })
    .insert_axis(Axis(1))
}

fn generate_noise(variance: f64, length: usize, mut rng: &mut impl Rng) -> Array2<f64> {
    let error_dist = Normal::new(0.0, variance).unwrap();
    let errors = (0..length)
        .map(|_| error_dist.sample(&mut rng))
        .collect::<Vec<_>>();
    Array2::from_shape_vec((length, 1), errors).unwrap()
}

fn func_to_learn(x: ArrayView2<f64>) -> Array2<f64> {
    let y = func(x);
    let mut rng = thread_rng();
    let noise = generate_noise(1.5, y.len(), &mut rng);
    y + noise
}

fn generate_data(
    data_size: usize,
    dim: usize,
    min: f64,
    max: f64,
    mut rng: &mut impl Rng,
) -> Array2<f64> {
    let uniform = Uniform::new(min, max);
    let data = (0..(data_size * dim))
        .map(|_| uniform.sample(&mut rng))
        .collect::<Vec<_>>();
    ArrayView2::from_shape((data_size, dim), &data)
        .unwrap()
        .to_owned()
}

fn mse(y: ArrayView2<f64>, y_hat: ArrayView2<f64>) -> f64 {
    (y.to_owned() - y_hat).map(|v| v.powi(2)).mean().unwrap()
}

fn main() {
    let mut rng = thread_rng();
    let x_train = generate_data(160, 3, -10.0, 10.0, &mut rng);
    let y_train = func_to_learn(x_train.view());
    dbg!(&y_train);

    let batch_size = 16;
    let mut network = Network::new(0.001);
    network.add_layer(Linear::new(x_train.ncols(), 4, batch_size, Relu));
    network.add_layer(Linear::new(4, y_train.ncols(), batch_size, Identity));

    let epochs = 3000;
    for epoch in 0..epochs {
        let mut has_processed = 0;
        let mut total_loss = 0.0;
        let mut n_iters = 0;
        print!("epoch {}: ", epoch);
        while has_processed < x_train.nrows() {
            print!("#");
            n_iters += 1;

            let x_train_batch = x_train
                .slice(s![has_processed..(has_processed + batch_size), ..])
                .to_owned();
            let y_train_batch = y_train
                .slice(s![has_processed..(has_processed + batch_size), ..])
                .to_owned();
            let y_hat = network.forward(x_train_batch);
            let loss = mse(y_train_batch.view(), y_hat.view());
            total_loss += loss;
            let error = (y_hat - y_train_batch) / batch_size as f64;
            network.backward(error);
            has_processed += batch_size;
        }
        let mean_loss = total_loss / n_iters as f64;
        println!(" mean loss: {}", mean_loss);
    }

    let x_test = generate_data(20, 3, -10.0, 10.0, &mut rng);
    let y_test = func(x_test.view());
    let y_pred = network.forward(x_test);
    println!("test error: {}", y_test - y_pred);
}
