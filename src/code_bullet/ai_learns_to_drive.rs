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

const CAR_HEIGHT: f64 = 50f64;
const CAR_WIDTH: f64 = 25f64;
const CAR_START_X: f64 = 87f64;
const CAR_START_Y: f64 = 573f64;

type BasicLine = ((f64, f64), (f64, f64));

const TRACK_LINES: [BasicLine; 46] = [
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

const REWARD_GATES: [BasicLine; 35] = [
    ((15f64, 539f64), (182f64, 532f64)),
    ((25f64, 371f64), (176f64, 419f64)),
    ((93f64, 221f64), (240f64, 333f64)),
    ((230f64, 118f64), (312f64, 275f64)),
    ((334f64, 83f64), (408f64, 238f64)),
    ((520f64, 73f64), (514f64, 202f64)),
    ((615f64, 72f64), (616f64, 213f64)),
    ((710f64, 84f64), (709f64, 219f64)),
    ((795f64, 85f64), (805f64, 215f64)),
    ((892f64, 94f64), (901f64, 239f64)),
    ((1012f64, 102f64), (1027f64, 239f64)),
    ((1114f64, 253f64), (1291f64, 225f64)),
    ((1071f64, 306f64), (1162f64, 436f64)),
    ((902f64, 396f64), (973f64, 503f64)),
    ((723f64, 474f64), (801f64, 575f64)),
    ((528f64, 585f64), (774f64, 591f64)),
    ((810f64, 587f64), (747f64, 750f64)),
    ((945f64, 576f64), (878f64, 714f64)),
    ((1051f64, 555f64), (1074f64, 690f64)),
    ((1120f64, 555f64), (1176f64, 659f64)),
    ((1216f64, 485f64), (1298f64, 589f64)),
    ((1363f64, 401f64), (1401f64, 516f64)),
    ((1564f64, 364f64), (1552f64, 497f64)),
    ((1765f64, 441f64), (1693f64, 541f64)),
    ((1892f64, 664f64), (1760f64, 683f64)),
    ((1903f64, 899f64), (1756f64, 810f64)),
    ((1640f64, 1099f64), (1591f64, 912f64)),
    ((1350f64, 1065f64), (1390f64, 912f64)),
    ((1098f64, 1066f64), (1194f64, 915f64)),
    ((847f64, 1060f64), (885f64, 929f64)),
    ((625f64, 1057f64), (680f64, 886f64)),
    ((434f64, 1056f64), (500f64, 912f64)),
    ((258f64, 1062f64), (326f64, 931f64)),
    ((97f64, 949f64), (241f64, 882f64)),
    ((56f64, 801f64), (211f64, 785f64)),
];

/* --- --- --- A.I. LEARNS TO DRIVE --- --- --- */

/// The goal is to drive as fast as possible around the track without ever touching the track edges.
///
/// The agent (a car) starts at the left most part of the track. For any given state the agent may
/// choose to accelerate forward, backward or neither and to rotate left, right or neither.
///
/// ## Observation
/// Space-Structure: `[12]`
///
/// | Index | Observation | Min | Max |
/// | --- | --- | --- | --- |
/// | `[0]` | Forward or Backward Velocity | `-1.0` | `1.0` |
/// | `[1]` | Right or Left Velocity | `-1.0` | `1.0` |
/// | `[2]` | Frontal Distance | `0.0` | `1.0` |
/// | `[3]` | Frontal Right First Distance | `0.0` | `1.0` |
/// | `[4]` | Frontal Right Second Distance | `0.0` | `1.0` |
/// | `[5]` | Frontal Right Third Distance | `0.0` | `1.0` |
/// | `[6]` | Rear Right Distance | `0.0` | `1.0` |
/// | `[7]` | Rear Distance | `0.0` | `1.0` |
/// | `[8]` | Rear Left Distance | `0.0` | `1.0` |
/// | `[9]` | Rear Left Third Distance | `0.0` | `1.0` |
/// | `[10]` | Rear Left Second Distance | `0.0` | `1.0` |
/// | `[11]` | Rear Left First Distance | `0.0` | `1.0` |
///
/// ## Actions
/// Space-Structure: `[2]`
///
/// | Index | Value | Action |
/// | --- | --- | --- |
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
    car_angle_in_degrees: f64,

    pub car_maximum_velocity: f64,
    pub car_sensor_distance: f64,

    last_collisions: Vec<Position2D>,
    last_reward_gate_touched: Option<usize>,

    pub show_sensor_lines: bool,
    pub show_track: bool,
    pub show_reward_gates: bool,
}

impl AiLearnsToDrive {
    fn calculate_collisions(
        &self,
        lines: &[BasicLine],
        previous_position: Position2D,
    ) -> Vec<(usize, Position2D)> {
        let current_transformations = Transformations2D {
            transformations: vec![
                Transformation2D::rotation(self.car_angle_in_degrees),
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
            lines
                .iter()
                .enumerate()
                .map(|(index, track_line)| {
                    let ((track_line_a_1, track_line_a_2), (track_line_b_1, track_line_b_2)) =
                        track_line;
                    (
                        index,
                        (
                            *position_tuple,
                            (
                                Position2D::with(*track_line_a_1, *track_line_a_2),
                                Position2D::with(*track_line_b_1, *track_line_b_2),
                            ),
                        ),
                    )
                })
                .collect::<Vec<(usize, ((Position2D, Position2D), (Position2D, Position2D)))>>()
                .into_iter()
        })
        .flatten()
        .map(|(index, compressed_args)| {
            (
                index,
                calculate_line_intersection_compressed_args(compressed_args),
            )
        })
        .filter(|(_, intersection)| intersection.is_some())
        .map(|(index, intersection)| (index, intersection.unwrap()))
        .collect()
    }

    fn calculate_track_line_collisions(&self, previous_position: Position2D) -> Vec<Position2D> {
        self.calculate_collisions(&TRACK_LINES, previous_position)
            .into_iter()
            .map(|(_, collision_position)| collision_position)
            .collect()
    }

    fn calculate_reward_gate_collisions(&self, previous_position: Position2D) -> Vec<usize> {
        self.calculate_collisions(&REWARD_GATES, previous_position)
            .into_iter()
            .map(|(index, _)| index)
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

    fn car_angle_in_radians(&self) -> f64 {
        gymnarium_base::math::degrees_to_radians(self.car_angle_in_degrees)
    }

    fn raycast_lines(&self) -> [(Position2D, Position2D); 10] {
        let rotation = Transformations2D {
            transformations: vec![Transformation2D::rotation(self.car_angle_in_degrees)],
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
                front_pos + (Vector2D::with(0f64, -1f64) * self.car_sensor_distance),
            ),
            // Front Right First
            (
                front_right_corner,
                front_right_corner
                    + (Vector2D::with(1f64, -5f64).normalized() * self.car_sensor_distance),
            ),
            // Front Right Second
            (
                front_right_corner,
                front_right_corner
                    + (Vector2D::with(1f64, -1f64).normalized() * self.car_sensor_distance),
            ),
            // Front Right Third
            (
                front_right_corner,
                front_right_corner
                    + (Vector2D::with(1f64, 0f64).normalized() * self.car_sensor_distance),
            ),
            // Back Right
            (
                back_right_corner,
                back_right_corner
                    + (Vector2D::with(1f64, 1f64).normalized() * self.car_sensor_distance),
            ),
            // Back
            (
                back_pos,
                back_pos + (Vector2D::with(0f64, 1f64) * self.car_sensor_distance),
            ),
            // Back Left
            (
                back_left_corner,
                back_left_corner
                    + (Vector2D::with(-1f64, 1f64).normalized() * self.car_sensor_distance),
            ),
            // Front Left Third
            (
                front_left_corner,
                front_left_corner
                    + (Vector2D::with(-1f64, 0f64).normalized() * self.car_sensor_distance),
            ),
            // Front Left Second
            (
                front_left_corner,
                front_left_corner
                    + (Vector2D::with(-1f64, -1f64).normalized() * self.car_sensor_distance),
            ),
            // Front Left First
            (
                front_left_corner,
                front_left_corner
                    + (Vector2D::with(-1f64, -5f64).normalized() * self.car_sensor_distance),
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
            car_angle_in_degrees: 0f64,

            car_sensor_distance: 750f64,
            car_maximum_velocity: 100f64,

            last_collisions: Vec::new(),
            last_reward_gate_touched: None,

            show_sensor_lines: false,
            show_track: true,
            show_reward_gates: false,
        }
    }
}

impl Environment<AiLearnsToDriveError, f64, (), AiLearnsToDriveStorage> for AiLearnsToDrive {
    fn action_space() -> ActionSpace {
        ActionSpace::simple_all(DimensionBoundaries::INTEGER(-1, 1), 2)
    }

    fn observation_space() -> ObservationSpace {
        ObservationSpace::simple(vec![
            DimensionBoundaries::FLOAT(-1f32, 1f32),
            DimensionBoundaries::FLOAT(-1f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
            DimensionBoundaries::FLOAT(0f32, 1f32),
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
        self.car_angle_in_degrees = 0f64;

        self.last_collisions = Vec::new();
        self.last_reward_gate_touched = None;

        Ok(self.state())
    }

    fn state(&self) -> EnvironmentState {
        let relative_velocity_vector = Position2D::zero()
            .vector_to(
                &(Position2D::zero() + self.car_velocity).transform(&Transformations2D {
                    transformations: vec![Transformation2D::rotation(-self.car_angle_in_degrees)],
                }),
            )
            .normalized()
            * self.car_velocity.length();
        let raycasted_points = self.raycast();
        // This manual indexing and long list of creating the state is there so that indexing problems may be discovered by the compiler
        EnvironmentState::simple(vec![
            DimensionValue::FLOAT(relative_velocity_vector.x as f32),
            DimensionValue::FLOAT(relative_velocity_vector.y as f32),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[0]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[1]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[2]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[3]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[4]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[5]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[6]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[7]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[8]).length()
                    / self.car_sensor_distance) as f32,
            ),
            DimensionValue::FLOAT(
                (self.car_position.vector_to(&raycasted_points[9]).length()
                    / self.car_sensor_distance) as f32,
            ),
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
        self.car_angle_in_degrees += 3.0f64 * rotation_action as f64;

        // ACCELERATE CAR
        self.car_velocity += Vector2D::with(
            self.car_angle_in_radians().sin(),
            -self.car_angle_in_radians().cos(),
        )
        .normalized()
            * (CAR_ACCELERATION * acceleration_action as f64);
        self.car_velocity *= 0.97f64;
        if self.car_velocity.length().abs() > self.car_maximum_velocity {
            self.car_velocity = self.car_velocity.normalized() * self.car_maximum_velocity;
        } else if self.car_velocity.length().abs() < (self.car_maximum_velocity / 1000f64) {
            self.car_velocity = Vector2D::zero();
        }

        // MOVE CAR
        let previous_position = self.car_position;
        self.car_position += self.car_velocity;

        // CHECK COLLISSIONS
        self.last_collisions = self.calculate_track_line_collisions(previous_position);

        // CHECK REWARD GATE
        let new_touched_reward_gate_indices =
            self.calculate_reward_gate_collisions(previous_position);

        let touched_new_reward_gate = if new_touched_reward_gate_indices.is_empty() {
            false
        } else if let Some(some_last_reward_gate_touched) = self.last_reward_gate_touched {
            let mut next_reward_gate = some_last_reward_gate_touched + 1;
            while next_reward_gate >= REWARD_GATES.len() {
                next_reward_gate -= REWARD_GATES.len();
            }
            if new_touched_reward_gate_indices.contains(&next_reward_gate) {
                self.last_reward_gate_touched = Some(new_touched_reward_gate_indices[0]);
                true
            } else {
                false
            }
        } else {
            self.last_reward_gate_touched = Some(new_touched_reward_gate_indices[0]);
            true
        };

        // CALCULATE REWARD
        let reward = -1f64
            + if self.last_collisions.is_empty() {
                0f64
            } else {
                -100f64
            }
            + if touched_new_reward_gate {
                100f64
            } else {
                0f64
            };

        Ok((self.state(), reward, !self.last_collisions.is_empty(), ()))
    }

    fn load(&mut self, data: AiLearnsToDriveStorage) -> Result<(), AiLearnsToDriveError> {
        self.car_velocity = data.car_velocity;
        self.car_position = data.car_position;
        self.car_angle_in_degrees = data.car_angle;
        self.last_collisions = data.last_collisions;
        Ok(())
    }

    fn store(&self) -> AiLearnsToDriveStorage {
        AiLearnsToDriveStorage {
            car_velocity: self.car_velocity,
            car_position: self.car_position,
            car_angle: self.car_angle_in_degrees,
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

        if self.show_reward_gates {
            REWARD_GATES
                .iter()
                .map(|((gate_a_x, gate_a_y), (gate_b_x, gate_b_y))| {
                    Geometry2D::line(
                        Position2D::with(*gate_a_x, *gate_a_y),
                        Position2D::with(*gate_b_x, *gate_b_y),
                    )
                    .line_or_border_color(Color::green())
                    .line_or_border_width(5f64)
                })
                .for_each(|gate_line| geometries.push(gate_line));
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
            .rotate_around_self(self.car_angle_in_degrees);
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
            let line_b_offset = line_b_start.y - line_b_slope * line_b_start.x;

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
            let line_b_offset = line_b_start.y - line_b_slope * line_b_start.x;

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

    #[test]
    fn test_one_horizontal_one_other_line_intersect() {
        // ((63f64, 355f64), (166f64, 187f64)),
        assert_eq!(
            calculate_line_intersection(
                (
                    Position2D::with(63f64, 355f64),
                    Position2D::with(166f64, 187f64)
                ),
                (
                    Position2D::with(75f64, 270f64),
                    Position2D::with(155f64, 270f64)
                ),
            ),
            Some(Position2D::with(115.11309523809524f64, 270f64))
        );
    }

    #[test]
    fn test_one_vertical_one_other_line_intersect() {
        // ((63f64, 355f64), (166f64, 187f64)),
        assert_eq!(
            calculate_line_intersection_compressed_args((
                (
                    Position2D::with(63f64, 355f64),
                    Position2D::with(166f64, 187f64)
                ),
                (
                    Position2D::with(99.5f64, 273.8f64),
                    Position2D::with(99.5f64, 323.8f64)
                ),
            )),
            Some(Position2D::with(99.5, 295.46601941747576))
        );
    }
}
