use ami::grad::{addition::add, matmul::matmul, mse::mse, relu::relu, Function, Tensor, Variable};
use ndarray::{s, Array, Array2, ArrayView2, Axis, Dimension, Zip};
use ndarray_rand::{
    rand::{thread_rng, Rng},
    rand_distr::{Distribution, Normal, Uniform},
    RandomExt,
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

fn update_parameter<D: Dimension>(p: &Variable<D>, lr: f32) {
    let mut buffer = Tensor::zeros(p.data().raw_dim());
    Zip::from(&mut buffer)
        .and(&*p.data())
        .and(&*p.gradient())
        .for_each(|buffer, d, g| *buffer = d - lr * g);
    *p.data_mut() = buffer;
}

#[test]
fn regression_against_noisy_function() {
    let mut rng = thread_rng();
    let x_min = -1.0;
    let x_max = 1.0;
    let x_train = generate_data(160, 3, x_min, x_max, &mut rng);
    let y_train = func_to_learn(x_train.view());

    let batch_size = 16;
    let w1 = Variable::new(Array::random((3, 4), Uniform::new(-1.0, 1.0))).requires_grad();
    let b1 = Variable::new(Array::random((4,), Uniform::new(-1.0, 1.0))).requires_grad();
    let w2 = Variable::new(Array::random((4, 1), Uniform::new(-1.0, 1.0))).requires_grad();
    let b2 = Variable::new(Array::random((1,), Uniform::new(-1.0, 1.0))).requires_grad();

    let lr = 1e-3;
    let epochs = 1000;
    for epoch in 0..epochs {
        let mut has_processed = 0;
        let mut total_loss = 0.0;
        let mut n_iters = 0;
        print!("epoch {}: ", epoch);
        while has_processed < x_train.nrows() {
            print!("#");
            n_iters += 1;

            let x_train_batch = Variable::new(
                x_train
                    .slice(s![has_processed..(has_processed + batch_size), ..])
                    .to_owned(),
            );
            let y_train_batch = Variable::new(
                y_train
                    .slice(s![has_processed..(has_processed + batch_size), ..])
                    .to_owned(),
            );

            let p1 = matmul(&x_train_batch, &w1);
            let a1 = add(&p1, &b1);
            let y1 = relu(&a1);
            let p2 = matmul(&y1, &w2);
            let a2 = add(&p2, &b2);
            // let y = identity(&a);
            let loss = mse(&a2, &y_train_batch);
            loss.forward();
            loss.backward();

            update_parameter(&w1, lr);
            update_parameter(&b1, lr);
            update_parameter(&w2, lr);
            update_parameter(&b2, lr);
            w1.zero_gradient();
            b1.zero_gradient();
            w2.zero_gradient();
            b2.zero_gradient();
            total_loss += loss.data()[()];
            has_processed += batch_size;
        }
        let mean_loss = total_loss / n_iters as f32;
        println!(" mean loss: {}", mean_loss);
    }

    let x_test = generate_data(20, 3, x_min, x_max, &mut rng);
    let y_test = func(x_test.view());
    let y1 = relu(&add(&matmul(&Variable::new(x_test), &w1), &b1));
    let y_pred = add(&matmul(&y1, &w2), &b2);
    y_pred.forward();
    println!("test pred: {}", y_pred.data().clone());
    println!("test error: {}", y_test - y_pred.data().clone());
}
