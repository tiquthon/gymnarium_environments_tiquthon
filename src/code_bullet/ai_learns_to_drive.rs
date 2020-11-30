//! # Code Bullet A.I. Learns to DRIVE
//!
//! These environments are all inspired or copied from the YouTube video
//! [A.I. Learns to DRIVE](https://www.youtube.com/watch?v=r428O_CMcpI).

use gymnarium_base::math::{Position2D, Size2D, Transformation2D, Transformations2D, Vector2D};
use gymnarium_base::space::{DimensionBoundaries, DimensionValue};
use gymnarium_base::{
    ActionSpace, AgentAction, Environment, EnvironmentState, ObservationSpace, Seed, ToActionMapper,
};

use gymnarium_visualisers_base::input::{Button, ButtonState, Input, Key};
use gymnarium_visualisers_base::{
    Color, DrawableEnvironment, Geometry2D, TwoDimensionalDrawableEnvironment, Viewport2D,
    Viewport2DModification,
};

use serde::{Deserialize, Serialize};

/* --- --- --- GENERAL --- --- --- */

#[derive(Debug)]
pub enum AiLearnsToDriveError {
    GivenActionDoesNotFitActionSpace,
}

impl std::fmt::Display for AiLearnsToDriveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GivenActionDoesNotFitActionSpace => {
                write!(f, "Given Action does not fit ActionSpace")
            }
        }
    }
}

impl std::error::Error for AiLearnsToDriveError {}

const CAR_ACCELERATION: f64 = 0.70f64;
const MAXIMUM_VELOCITY: f64 = 100f64;

const CAR_SENSOR_DISTANCE: f64 = 750.0f64;

const CAR_HEIGHT: f64 = 50f64;
const CAR_WIDTH: f64 = 25f64;
const CAR_START_X: f64 = 87f64;
const CAR_START_Y: f64 = 573f64;

const TRACK_LINES: [((f64, f64), (f64, f64)); 46] = [
    // Segment 1:
    ((42f64, 580f64), (63f64, 355f64)),
    ((143f64, 369f64), (132f64, 567f64)),
    // Segment 2:
    ((63f64, 355f64), (166f64, 187f64)),
    ((238f64, 261f64), (143f64, 369f64)),
    // Segment 3:
    ((166f64, 187f64), (456f64, 88f64)),
    ((476f64, 168f64), (238f64, 261f64)),
    // Segment 4:
    ((456f64, 88f64), (1075f64, 119f64)),
    ((1139f64, 222f64), (476f64, 168f64)),
    // Segment 5:
    ((1075f64, 119f64), (1212f64, 138f64)),
    ((1139f64, 222f64), (1139f64, 222f64)),
    // Segment 6:
    ((1212f64, 138f64), (1267f64, 209f64)),
    ((1139f64, 222f64), (1139f64, 222f64)),
    // Segment 7:
    ((1267f64, 209f64), (1264f64, 321f64)),
    ((1146f64, 268f64), (1139f64, 222f64)),
    // Segment 8:
    ((1264f64, 321f64), (1182f64, 404f64)),
    ((1146f64, 268f64), (1146f64, 268f64)),
    // Segment 9:
    ((1182f64, 404f64), (1054f64, 433f64)),
    ((1054f64, 335f64), (1146f64, 268f64)),
    // Segment 10:
    ((1054f64, 433f64), (710f64, 600f64)),
    ((622f64, 537f64), (1054f64, 335f64)),
    // Segment 11:
    ((710f64, 600f64), (710f64, 600f64)),
    ((527f64, 612f64), (622f64, 537f64)),
    // Segment 12:
    ((710f64, 600f64), (710f64, 600f64)),
    ((526f64, 714f64), (527f64, 612f64)),
    // Segment 13:
    ((710f64, 600f64), (710f64, 600f64)),
    ((643f64, 735f64), (526f64, 714f64)),
    // Segment 14:
    ((710f64, 600f64), (1121f64, 564f64)),
    ((1158f64, 650f64), (643f64, 735f64)),
    // Segment 15:
    ((1121f64, 564f64), (1348f64, 412f64)),
    ((1385f64, 500f64), (1158f64, 650f64)),
    // Segment 16:
    ((1348f64, 412f64), (1557f64, 379f64)),
    ((1558f64, 477f64), (1385f64, 500f64)),
    // Segment 17:
    ((1557f64, 379f64), (1742f64, 456f64)),
    ((1696f64, 523f64), (1558f64, 477f64)),
    // Segment 18:
    ((1742f64, 456f64), (1874f64, 695f64)),
    ((1797f64, 714f64), (1696f64, 523f64)),
    // Segment 19:
    ((1874f64, 695f64), (1864f64, 927f64)),
    ((1762f64, 862f64), (1797f64, 714f64)),
    // Segment 20:
    ((1864f64, 927f64), (1627f64, 1037f64)),
    ((1601f64, 947f64), (1762f64, 862f64)),
    // Segment 21:
    ((1627f64, 1037f64), (210f64, 1034f64)),
    ((276f64, 940f64), (1601f64, 947f64)),
    // Segment 22:
    ((210f64, 1034f64), (119f64, 958f64)),
    ((198f64, 886f64), (276f64, 940f64)),
    // Segment 23:
    ((119f64, 958f64), (42f64, 580f64)),
    ((132f64, 567f64), (198f64, 886f64)),
];

/* --- --- --- A.I. LEARNS TO DRIVE --- --- --- */

/// The goal is to drive as fast as possible around the track without ever touching the track edges.
///
/// The agent (a car) starts at the left most part of the track. For any given state the agent may
/// choose to accelerate forward, backward or neitzer and to rotate left, right or neither.
///
/// ## Observation
/// Space-Structure: `[8]`
///
/// | Index | Observation | Min | Max |
/// | --- | --- | --- | --- |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
/// | `[0]` | Frontal Distance | `0.0` | `750.0` |
///
/// ## Actions
/// Space-Structure: `[2]`
///
/// | Index | Value | Action |
/// | `[0]` | `-1` | Accelerate backwards |
/// | `[0]` | `0` | Don't accelerate |
/// | `[0]` | `1` | Accelerate forwards |
/// | `[1]` | `-1` | Rotate left |
/// | `[1]` | `0` | Don't rotate |
/// | `[1]` | `1` | Rotate right |
///
/// ## Reward
///
/// TODO
///
/// ## Starting State
/// The position of the car is always assigned to `(87|573)`.
///
/// The velocity of the car is always assigned to `0`.
///
/// ## Episode Termination
///
/// TODO
///
pub struct AiLearnsToDrive {
    car_velocity: Vector2D,
    car_position: Position2D,
    car_angle: f64,

    last_collisions: Vec<Position2D>,

    show_sensor_lines: bool,
    show_track: bool,
}

impl AiLearnsToDrive {
    pub fn sensor_lines_visible(&mut self, visible: bool) {
        self.show_sensor_lines = visible;
    }

    pub fn track_visible(&mut self, visible: bool) {
        self.show_track = visible;
    }

    fn calculate_collisions(&mut self, previous_position: Position2D) -> Vec<Position2D> {
        let current_transformations = Transformations2D {
            transformations: vec![
                Transformation2D::rotation(self.car_angle),
                Transformation2D::translation(Position2D::zero().vector_to(&self.car_position)),
            ],
        };
        [
            (previous_position, self.car_position),
            (
                Self::car_corners().0.transform(&current_transformations),
                Self::car_corners().1.transform(&current_transformations),
            ),
            (
                Self::car_corners().1.transform(&current_transformations),
                Self::car_corners().2.transform(&current_transformations),
            ),
            (
                Self::car_corners().2.transform(&current_transformations),
                Self::car_corners().3.transform(&current_transformations),
            ),
            (
                Self::car_corners().3.transform(&current_transformations),
                Self::car_corners().0.transform(&current_transformations),
            ),
        ]
        .iter()
        .map(|position_tuple| {
            TRACK_LINES
                .iter()
                .map(|track_line| {
                    let ((track_line_a_1, track_line_a_2), (track_line_b_1, track_line_b_2)) =
                        track_line;
                    (
                        *position_tuple,
                        (
                            Position2D::with(*track_line_a_1, *track_line_a_2),
                            Position2D::with(*track_line_b_1, *track_line_b_2),
                        ),
                    )
                })
                .collect::<Vec<((Position2D, Position2D), (Position2D, Position2D))>>()
                .into_iter()
        })
        .flatten()
        .map(calculate_line_intersection_compressed_args)
        .filter(|intersection| intersection.is_some())
        .map(|intersection| intersection.unwrap())
        .collect()
    }

    fn car_corners() -> (Position2D, Position2D, Position2D, Position2D) {
        (
            Position2D::with(-CAR_WIDTH / 2f64, CAR_HEIGHT / 2f64),
            Position2D::with(CAR_WIDTH / 2f64, CAR_HEIGHT / 2f64),
            Position2D::with(CAR_WIDTH / 2f64, -CAR_HEIGHT / 2f64),
            Position2D::with(-CAR_WIDTH / 2f64, -CAR_HEIGHT / 2f64),
        )
    }

    fn raycast_lines(&self) -> [(Position2D, Position2D); 10] {
        let rotation = Transformations2D {
            transformations: vec![Transformation2D::rotation(self.car_angle)],
        };
        let car_pos_vec = Position2D::zero().vector_to(&self.car_position);

        let front_pos = Position2D::with(0f64, -CAR_HEIGHT / 2f64);
        let front_right_corner = Position2D::with(CAR_WIDTH / 2f64, -CAR_HEIGHT / 2f64);
        let front_left_corner = Position2D::with(-CAR_WIDTH / 2f64, -CAR_HEIGHT / 2f64);
        let back_right_corner = Position2D::with(CAR_WIDTH / 2f64, CAR_HEIGHT / 2f64);
        let back_pos = Position2D::with(0f64, CAR_HEIGHT / 2f64);
        let back_left_corner = Position2D::with(-CAR_WIDTH / 2f64, CAR_HEIGHT / 2f64);

        let m = [
            // Front
            (
                front_pos,
                front_pos + (Vector2D::with(0f64, -1f64) * CAR_SENSOR_DISTANCE),
            ),
            // Front Right First
            (
                front_right_corner,
                front_right_corner
                    + (Vector2D::with(1f64, -5f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Front Right Second
            (
                front_right_corner,
                front_right_corner
                    + (Vector2D::with(1f64, -1f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Front Right Third
            (
                front_right_corner,
                front_right_corner
                    + (Vector2D::with(1f64, 0f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Back Right
            (
                back_right_corner,
                back_right_corner + (Vector2D::with(1f64, 1f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Back
            (
                back_pos,
                back_pos + (Vector2D::with(0f64, 1f64) * CAR_SENSOR_DISTANCE),
            ),
            // Back Left
            (
                back_left_corner,
                back_left_corner + (Vector2D::with(-1f64, 1f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Front Left Third
            (
                front_left_corner,
                front_left_corner
                    + (Vector2D::with(-1f64, 0f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Front Left Second
            (
                front_left_corner,
                front_left_corner
                    + (Vector2D::with(-1f64, -1f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
            // Front Left First
            (
                front_left_corner,
                front_left_corner
                    + (Vector2D::with(-1f64, -5f64).normalized() * CAR_SENSOR_DISTANCE),
            ),
        ]
        .iter()
        .map(|(posa, posb)| (posa.transform(&rotation), posb.transform(&rotation)))
        .map(|(posa, posb)| (posa + car_pos_vec, posb + car_pos_vec))
        .collect::<Vec<(Position2D, Position2D)>>();
        [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9]]
    }

    fn raycast(&self) -> [Position2D; 10] {
        let raycasted_points = self
            .raycast_lines()
            .iter()
            .map(|(start_pos, end_pos)| {
                TRACK_LINES
                    .iter()
                    .filter_map(|((line_a_x, line_a_y), (line_b_x, line_b_y))| {
                        calculate_line_intersection(
                            (
                                Position2D::with(*line_a_x, *line_a_y),
                                Position2D::with(*line_b_x, *line_b_y),
                            ),
                            (*start_pos, *end_pos),
                        )
                    })
                    .fold_first(|a, b| {
                        if start_pos.vector_to(&a).length() < start_pos.vector_to(&b).length() {
                            a
                        } else {
                            b
                        }
                    })
                    .unwrap_or(*end_pos)
            })
            .collect::<Vec<Position2D>>();
        [
            raycasted_points[0],
            raycasted_points[1],
            raycasted_points[2],
            raycasted_points[3],
            raycasted_points[4],
            raycasted_points[5],
            raycasted_points[6],
            raycasted_points[7],
            raycasted_points[8],
            raycasted_points[9],
        ]
    }
}

impl Default for AiLearnsToDrive {
    fn default() -> Self {
        Self {
            car_velocity: Vector2D::zero(),
            car_position: Position2D::with(CAR_START_X, CAR_START_Y),
            car_angle: 0f64,

            last_collisions: Vec::new(),

            show_sensor_lines: false,
            show_track: true,
        }
    }
}

impl Environment<AiLearnsToDriveError, (), AiLearnsToDriveStorage> for AiLearnsToDrive {
    fn action_space() -> ActionSpace {
        ActionSpace::simple(vec![
            DimensionBoundaries::INTEGER(-1, 1),
            DimensionBoundaries::INTEGER(-1, 1),
        ])
    }

    fn observation_space() -> ObservationSpace {
        ObservationSpace::simple(vec![
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
            DimensionBoundaries::FLOAT(0f32, CAR_SENSOR_DISTANCE as f32),
        ])
    }

    fn suggested_episode_steps_count() -> Option<u128> {
        None
    }

    fn reseed(&mut self, _random_seed: Option<Seed>) -> Result<(), AiLearnsToDriveError> {
        Ok(())
    }

    fn reset(&mut self) -> Result<EnvironmentState, AiLearnsToDriveError> {
        self.car_velocity = Vector2D::zero();
        self.car_position = Position2D::with(CAR_START_X, CAR_START_Y);
        self.car_angle = 0f64;
        Ok(self.state())
    }

    fn state(&self) -> EnvironmentState {
        let raycasted_points = self.raycast();
        EnvironmentState::simple(vec![
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[0]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[1]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[2]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[3]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[4]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[5]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[6]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[7]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[8]).length() as f32),
            DimensionValue::FLOAT(self.car_position.vector_to(&raycasted_points[9]).length() as f32),
        ])
    }

    fn step(
        &mut self,
        action: &AgentAction,
    ) -> Result<(EnvironmentState, f64, bool, ()), AiLearnsToDriveError> {
        if !Self::action_space().contains(action) {
            return Err(AiLearnsToDriveError::GivenActionDoesNotFitActionSpace);
        }

        // EXTRACT ACTION VALUES
        let acceleration_action = action[&[0]].expect_integer();
        let rotation_action = action[&[1]].expect_integer();

        // ROTATE CAR
        self.car_angle += 0.1f64 * rotation_action as f64;

        // ACCELERATE CAR
        self.car_velocity += Vector2D::with((self.car_angle).sin(), -(self.car_angle).cos())
            .normalized()
            * (CAR_ACCELERATION * acceleration_action as f64);
        self.car_velocity *= 0.97f64;
        if self.car_velocity.length().abs() > MAXIMUM_VELOCITY {
            self.car_velocity = self.car_velocity.normalized() * MAXIMUM_VELOCITY;
        } else if self.car_velocity.length().abs() < (MAXIMUM_VELOCITY / 1000f64) {
            self.car_velocity = Vector2D::zero();
        }

        // MOVE CAR
        let previous_position = self.car_position;
        self.car_position += self.car_velocity;

        // CHECK COLLISSIONS
        self.last_collisions = self.calculate_collisions(previous_position);

        Ok((self.state(), 0f64, !self.last_collisions.is_empty(), ()))
    }

    fn load(&mut self, data: AiLearnsToDriveStorage) -> Result<(), AiLearnsToDriveError> {
        self.car_velocity = data.car_velocity;
        self.car_position = data.car_position;
        self.car_angle = data.car_angle;
        self.last_collisions = data.last_collisions;
        Ok(())
    }

    fn store(&self) -> AiLearnsToDriveStorage {
        AiLearnsToDriveStorage {
            car_velocity: self.car_velocity,
            car_position: self.car_position,
            car_angle: self.car_angle,
            last_collisions: self.last_collisions.clone(),
        }
    }

    fn close(&mut self) -> Result<(), AiLearnsToDriveError> {
        Ok(())
    }
}

impl DrawableEnvironment for AiLearnsToDrive {
    fn suggested_rendered_steps_per_second() -> Option<f64> {
        Some(60f64)
    }
}

impl TwoDimensionalDrawableEnvironment<AiLearnsToDriveError> for AiLearnsToDrive {
    fn draw_two_dimensional(&self) -> Result<Vec<Geometry2D>, AiLearnsToDriveError> {
        let mut geometries = Vec::new();

        // DRAW TRACK
        if self.show_track {
            let mut track_segments = Vec::new();
            let mut track_lines_index = 0;
            while track_lines_index < TRACK_LINES.len() {
                let (
                    (track_line_a_start_x, track_line_a_start_y),
                    (track_line_a_end_x, track_line_a_end_y),
                ) = TRACK_LINES[track_lines_index];
                let (
                    (track_line_b_start_x, track_line_b_start_y),
                    (track_line_b_end_x, track_line_b_end_y),
                ) = TRACK_LINES[track_lines_index + 1];
                track_segments.push(
                    Geometry2D::polygon(vec![
                        Position2D::with(track_line_a_start_x, track_line_a_start_y),
                        Position2D::with(track_line_a_end_x, track_line_a_end_y),
                        Position2D::with(track_line_b_start_x, track_line_b_start_y),
                        Position2D::with(track_line_b_end_x, track_line_b_end_y),
                    ])
                    .fill_color(Color::with(160, 160, 160, 255)),
                );
                track_lines_index += 2;
            }
            let track = Geometry2D::group(track_segments);
            geometries.push(track);
        }

        // DRAW CAR
        let chassis = Geometry2D::polygon(vec![
            Self::car_corners().0,
            Self::car_corners().1,
            Self::car_corners().2,
            Self::car_corners().3,
        ])
        .fill_color(if self.last_collisions.is_empty() {
            Color::black()
        } else {
            Color::red()
        });
        let window =
            Geometry2D::rectangle(Position2D::with(0f64, -8f64), Size2D::with(19f64, 7f64))
                .fill_color(Color::white());
        let left_light =
            Geometry2D::rectangle(Position2D::with(-7.5f64, -23f64), Size2D::with(6f64, 4f64))
                .fill_color(Color::with(134, 139, 104, 255));
        let right_light =
            Geometry2D::rectangle(Position2D::with(7.5f64, -23f64), Size2D::with(6f64, 4f64))
                .fill_color(Color::with(134, 139, 104, 255));
        let car = Geometry2D::group(vec![chassis, window, left_light, right_light])
            .move_to(&Position2D::with(self.car_position.x, self.car_position.y))
            .rotate_around_self(self.car_angle);
        geometries.push(car);

        // DRAW RAYCAST
        if self.show_sensor_lines {
            let raycast_lines = Geometry2D::group(
                self.raycast_lines()
                    .iter()
                    .map(|(start, end)| Geometry2D::line(*start, *end))
                    .collect::<Vec<Geometry2D>>(),
            )
            .line_or_border_color(Color::white());
            geometries.push(raycast_lines);

            let raycast_points = Geometry2D::group(
                self.raycast()
                    .iter()
                    .map(|position| Geometry2D::circle(*position, 7f64))
                    .collect::<Vec<Geometry2D>>(),
            )
            .fill_color(Color::white());
            geometries.push(raycast_points);
        }

        Ok(geometries)
    }

    fn preferred_view(&self) -> Option<(Viewport2D, Viewport2DModification)> {
        Some((
            Viewport2D::with(
                Position2D::with(960f64, 540f64),
                Size2D::with(1920f64, 1080f64),
            )
            .flipped_y_axis(true),
            Viewport2DModification::KeepAspectRatio,
        ))
    }

    fn preferred_background_color(&self) -> Option<Color> {
        Some(Color::with(100, 100, 100, 255))
    }
}

#[derive(Serialize, Deserialize)]
pub struct AiLearnsToDriveStorage {
    car_velocity: Vector2D,
    car_position: Position2D,
    car_angle: f64,
    last_collisions: Vec<Position2D>,
}

#[derive(Default)]
pub struct AiLearnsToDriveInputToActionMapper {
    up_arrow_pressed: bool,
    down_arrow_pressed: bool,
    left_arrow_pressed: bool,
    right_arrow_pressed: bool,
}

impl ToActionMapper<Vec<Input>, AiLearnsToDriveError> for AiLearnsToDriveInputToActionMapper {
    fn map(&mut self, input: &Vec<Input>) -> Result<AgentAction, AiLearnsToDriveError> {
        for input_element in input {
            if let Input::Button(button_args) = input_element {
                if let Button::Keyboard(key) = button_args.button {
                    if Key::Up == key {
                        self.up_arrow_pressed = button_args.state == ButtonState::Press;
                    } else if Key::Down == key {
                        self.down_arrow_pressed = button_args.state == ButtonState::Press;
                    } else if Key::Left == key {
                        self.left_arrow_pressed = button_args.state == ButtonState::Press;
                    } else if Key::Right == key {
                        self.right_arrow_pressed = button_args.state == ButtonState::Press;
                    }
                }
            }
        }
        Ok(AgentAction::simple(vec![
            DimensionValue::from(if self.up_arrow_pressed && !self.down_arrow_pressed {
                1
            } else if !self.up_arrow_pressed && self.down_arrow_pressed {
                -1
            } else {
                0
            }),
            DimensionValue::from(if self.right_arrow_pressed && !self.left_arrow_pressed {
                1
            } else if !self.right_arrow_pressed && self.left_arrow_pressed {
                -1
            } else {
                0
            }),
        ]))
    }
}

/* --- --- --- UTILITY --- --- --- */

fn calculate_line_intersection_compressed_args(
    lines: ((Position2D, Position2D), (Position2D, Position2D)),
) -> Option<Position2D> {
    calculate_line_intersection(lines.0, lines.1)
}

fn calculate_line_intersection(
    line_a: (Position2D, Position2D),
    line_b: (Position2D, Position2D),
) -> Option<Position2D> {
    let (line_a_start, line_a_end) = line_a;
    let (line_b_start, line_b_end) = line_b;

    let line_a_delta_x = line_a_end.x - line_a_start.x;
    let line_a_delta_y = line_a_end.y - line_a_start.y;
    let line_b_delta_x = line_b_end.x - line_b_start.x;
    let line_b_delta_y = line_b_end.y - line_b_start.y;

    let (intersection_x, intersection_y) = if line_a_delta_x.abs() == 0f64 {
        if line_b_delta_x.abs() == 0f64 {
            return None;
        } else {
            let line_b_slope = line_b_delta_y / line_b_delta_x;
            let line_b_offset = line_b_start.y - line_b_slope * line_b_start.y;

            let intersection_x = line_a_start.x;
            let intersection_y = line_b_slope * intersection_x + line_b_offset;
            (intersection_x, intersection_y)
        }
    } else if line_b_delta_x.abs() == 0f64 {
        if line_a_delta_x.abs() == 0f64 {
            return None;
        } else {
            let line_a_slope = line_a_delta_y / line_a_delta_x;
            let line_a_offset = line_a_start.y - line_a_slope * line_a_start.x;

            let intersection_x = line_b_start.x;
            let intersection_y = line_a_slope * intersection_x + line_a_offset;
            (intersection_x, intersection_y)
        }
    } else if line_a_delta_y.abs() == 0f64 {
        if line_b_delta_y.abs() == 0f64 {
            return None;
        } else {
            let line_b_slope = line_b_delta_y / line_b_delta_x;
            let line_b_offset = line_b_start.y - line_b_slope * line_b_start.y;

            let intersection_x = (line_a_start.y - line_b_offset) / line_b_slope;
            let intersection_y = line_a_start.y;
            (intersection_x, intersection_y)
        }
    } else if line_b_delta_y.abs() == 0f64 {
        if line_a_delta_y.abs() == 0f64 {
            return None;
        } else {
            let line_a_slope = line_a_delta_y / line_a_delta_x;
            let line_a_offset = line_a_start.y - line_a_slope * line_a_start.x;

            let intersection_x = (line_b_start.y - line_a_offset) / line_a_slope;
            let intersection_y = line_b_start.y;
            (intersection_x, intersection_y)
        }
    } else {
        let line_a_slope = line_a_delta_y / line_a_delta_x;
        let line_b_slope = line_b_delta_y / line_b_delta_x;

        let line_a_offset = line_a_start.y - line_a_slope * line_a_start.x;
        let line_b_offset = line_b_start.y - line_b_slope * line_b_start.x;

        if (line_a_slope - line_b_slope) == 0f64 {
            return None;
        }

        let intersection_x = (line_b_offset - line_a_offset) / (line_a_slope - line_b_slope);
        let intersection_y = line_a_slope * intersection_x + line_a_offset;

        (intersection_x, intersection_y)
    };

    if intersection_x > line_a_start.x.max(line_a_end.x)
        || intersection_x < line_a_start.x.min(line_a_end.x)
        || intersection_y > line_a_start.y.max(line_a_end.y)
        || intersection_y < line_a_start.y.min(line_a_end.y)
        || intersection_x > line_b_start.x.max(line_b_end.x)
        || intersection_x < line_b_start.x.min(line_b_end.x)
        || intersection_y > line_b_start.y.max(line_b_end.y)
        || intersection_y < line_b_start.y.min(line_b_end.y)
    {
        None
    } else {
        Some(Position2D::with(intersection_x, intersection_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossing_lines_with_two_diagonal_lines_intersect() {
        assert_eq!(
            calculate_line_intersection(
                (Position2D::with(-1f64, -1f64), Position2D::with(1f64, 1f64)),
                (Position2D::with(-1f64, 1f64), Position2D::with(1f64, -1f64)),
            ),
            Some(Position2D::zero())
        );
    }

    #[test]
    fn test_crossing_lines_with_vertical_and_horizontal_line_intersect() {
        assert_eq!(
            calculate_line_intersection(
                (Position2D::with(-1f64, 0f64), Position2D::with(1f64, 0f64)),
                (Position2D::with(0f64, 1f64), Position2D::with(0f64, -1f64)),
            ),
            Some(Position2D::zero())
        );
    }

    #[test]
    fn test_parallel_lines_dont_intersect() {
        assert_eq!(
            calculate_line_intersection(
                (Position2D::with(-1f64, 0f64), Position2D::with(1f64, 0f64)),
                (Position2D::with(-1f64, 1f64), Position2D::with(1f64, 1f64)),
            ),
            None
        );
    }
}
