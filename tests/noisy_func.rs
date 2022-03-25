use ami::{
    data::{DataLoader, SequentialSampler},
    grad::{mse, Function, Variable},
    model::{Chainable, Input, Linear, Model, Relu},
    optimizer::GradientDescent,
    sequential,
};
use ndarray::{Array2, ArrayView2, Axis};
use ndarray_rand::{
    rand::{thread_rng, Rng},
    rand_distr::{Distribution, Normal, Uniform},
};

// x: (batch_size, input_size)
fn func(x: ArrayView2<f32>) -> Array2<f32> {
    x.map_axis(Axis(1), |v| {
        2.0 + 5.0 * v[[0]] + -3.0 * v[[1]].sin() + 4.0 * v[[2]].powi(2)
    })
    .insert_axis(Axis(1))
}

fn generate_noise(variance: f32, length: usize, mut rng: &mut impl Rng) -> Array2<f32> {
    let error_dist = Normal::new(0.0, variance).unwrap();
    let errors = (0..length)
        .map(|_| error_dist.sample(&mut rng))
        .collect::<Vec<_>>();
    Array2::from_shape_vec((length, 1), errors).unwrap()
}

fn func_to_learn(x: ArrayView2<f32>) -> Array2<f32> {
    let y = func(x);
    let mut rng = thread_rng();
    let noise = generate_noise(1.5, y.len(), &mut rng);
    y + noise
}

fn generate_data(
    data_size: usize,
    dim: usize,
    min: f32,
    max: f32,
    mut rng: &mut impl Rng,
) -> Array2<f32> {
    let uniform = Uniform::new(min, max);
    let data = (0..(data_size * dim))
        .map(|_| uniform.sample(&mut rng))
        .collect::<Vec<_>>();
    ArrayView2::from_shape((data_size, dim), &data)
        .unwrap()
        .to_owned()
}

#[test]
fn regression_against_noisy_function() {
    let mut rng = thread_rng();
    let x_min = -1.0;
    let x_max = 1.0;
    let x_train = generate_data(160, 3, x_min, x_max, &mut rng);
    let y_train = func_to_learn(x_train.view());

    let mut loader = DataLoader::new(SequentialSampler::new(x_train.nrows()), x_train, y_train);

    let batch_size = 16;
    let model = sequential!(Input, Linear::new(3, 4), Relu::new(), Linear::new(4, 1));

    let lr = 1e-3;
    let optimizer = GradientDescent::new(lr);

    let epochs = 1000;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        print!("epoch {}: ", epoch);
        for (input, target) in loader.batch(batch_size) {
            print!("#");

            let input = Variable::new(input);
            let target = Variable::new(target);

            let network = model.forward(input);
            let loss = mse(&network, &target);
            loss.forward();
            loss.backward();
            model.update_parameters(&optimizer);
            model.zero_gradient();

            total_loss += loss.data()[()];
        }
        println!(" total loss: {}", total_loss);
    }

    let x_test = generate_data(20, 3, x_min, x_max, &mut rng);
    let y_test = func(x_test.view());
    let y_pred = model.forward(Variable::new(x_test));
    y_pred.forward();
    println!("test pred: {}", y_pred.data().clone());
    println!("test error: {}", y_test - y_pred.data().clone());
}
