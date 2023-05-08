use bincode::serialize_into;
use plotters::{
    prelude::*,
    style::{AsRelative, WHITE},
};
use std::{collections::VecDeque, fs::File, path::Path};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PlotError {
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
}

pub struct Plotter {
    losses: VecDeque<(f32, f32, f32)>,
}

impl Plotter {
    pub const MAX_LOSSES: usize = 1024 * 1024;

    pub fn new() -> Self {
        Self {
            losses: VecDeque::new(),
        }
    }

    pub fn add_loss(&mut self, loss: (f32, f32, f32)) {
        if self.losses.len() == Self::MAX_LOSSES {
            self.losses.pop_front();
        }

        self.losses.push_back(loss);
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), PlotError> {
        let file = File::create(path)?;
        serialize_into(file, &self.losses)?;
        Ok(())
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<Self, PlotError> {
        let file = File::open(path)?;
        let losses = bincode::deserialize_from(file)?;
        Ok(Self { losses })
    }

    pub fn draw_plot(&mut self, path: impl AsRef<Path>) {
        draw_loss_plot(self.losses.make_contiguous(), path);
    }
}

fn draw_loss_plot(losses: &[(f32, f32, f32)], path: impl AsRef<Path>) {
    let root = SVGBackend::new(path.as_ref(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let len = losses.len();
    let max_loss = *losses
        .iter()
        .map(|(_, _, loss)| loss)
        .max_by(|a, b| f32::total_cmp(a, b))
        .unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Losses", ("sans-serif", 5.percent_height()))
        .set_label_area_size(LabelAreaPosition::Left, 8.percent())
        .set_label_area_size(LabelAreaPosition::Bottom, 4.percent())
        .margin((1).percent())
        .build_cartesian_2d(0..len, 0f32..max_loss)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Epochs")
        .y_desc("Loss")
        .draw()
        .unwrap();

    {
        let color = Palette99::pick(0).mix(0.9);
        chart
            .draw_series(LineSeries::new(
                losses
                    .iter()
                    .enumerate()
                    .map(|(epoch, &(_, _, loss))| (epoch, loss)),
                color.stroke_width(3),
            ))
            .unwrap()
            .label("Total Loss")
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    {
        let color = Palette99::pick(1).mix(0.9);
        chart
            .draw_series(LineSeries::new(
                losses
                    .iter()
                    .enumerate()
                    .map(|(epoch, &(loss, _, _))| (epoch, loss)),
                color.stroke_width(3),
            ))
            .unwrap()
            .label("Value Loss")
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    {
        let color = Palette99::pick(2).mix(0.9);
        chart
            .draw_series(LineSeries::new(
                losses
                    .iter()
                    .enumerate()
                    .map(|(epoch, &(_, loss, _))| (epoch, loss)),
                color.stroke_width(3),
            ))
            .unwrap()
            .label("Policy Loss")
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], color.filled()));
    }

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}
