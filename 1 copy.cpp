#include <Eigen/Eigen> //3.3.4
#include <iostream>
#include <iomanip>
#include <cmath>
#include "math.h"
#include <chrono>
#include "flight_data.h"
#include "nn.h"
// Hexsoon Edu 450 data
static constexpr double Gz = -9.81;
static constexpr double motor_tau = 0.0125;
static constexpr double hte = 0.6074674579665363; // hover thrust estimate
static constexpr double remaining_thrust = 1 - hte;
constexpr double integrator_dt = 0.005;
constexpr double motor_tau_inv = 1.0 / motor_tau;
constexpr double sim_dt = 0.008;
constexpr double motor_constant = 7.84e-6;
constexpr double rotor_drag_coef = 0.000175;
constexpr double mass = 1.6;
constexpr double arm_l = 0.225;
const Eigen::Vector<double, 4> min_rpm{100, 100, 100, 100}; // min rpm when quadcopter is armed
const Eigen::Vector<double, 3> _gz{0.0, 0.0, Gz};
const Eigen::Vector<double, 3> rotor_drag{4.5, 4.5, 7};
const Eigen::Matrix<double, 3, 3> J = mass / 12.0 * arm_l * arm_l * rotor_drag.asDiagonal(); // inertia matrix
const Eigen::Matrix<double, 3, 3> J_inv = J.inverse();
const Eigen::Matrix<double, 3, 4> t_BM_ = arm_l * sqrt(0.5) *
                                          (Eigen::Matrix<double, 3, 4>() << 1, -1, -1, 1, -1, -1, 1, 1, 0, 0, 0, 0).finished(); // thrust to moment matrix
const Eigen::Matrix<double, 4, 4> B_allocation_ = (Eigen::Matrix<double, 4, 4>() << Eigen::Vector<double, 4>::Ones().transpose(), t_BM_.topRows<2>(),
                                                   rotor_drag_coef *Eigen::Vector<double, 4>(1, -1, 1, -1).transpose())
                                                      .finished(); // Used to generate thrust equations (see torque matrix eq. )
const Eigen::Matrix<double, 4, 4> B_allocation_inv_ = B_allocation_.inverse();
class State
{
public:
    /**
     * @brief Indexes for the state vector
     */
    enum IDX : int
    {
        // position
        POS = 0,
        POSX = 0,
        POSY = 1,
        POSZ = 2,
        NPOS = 3,
        // quaternion
        ATT = 3,
        ATTW = 3,
        ATTX = 4,
        ATTY = 5,
        ATTZ = 6,
        NATT = 4,
        // linear velocity
        VEL = 7,
        VELX = 7,
        VELY = 8,
        VELZ = 9,
        NVEL = 3,
        // body rate
        OME = 10,
        OMEX = 10,
        OMEY = 11,
        OMEZ = 12,
        NOME = 3,
        // linear acceleration
        ACC = 13,
        ACCX = 13,
        ACCY = 14,
        ACCZ = 15,
        NACC = 3,
        // angular acceleration
        TAU = 16,
        TAUX = 16,
        TAUY = 17,
        TAUZ = 18,
        NTAU = 3,
        //
        BOME = 19,
        BOMEX = 19,
        BOMEY = 20,
        BOMEZ = 21,
        NBOME = 3,
        //
        BACC = 22,
        BACCX = 22,
        BACCY = 23,
        BACCZ = 24,
        NBACC = 3,
        //
        SIZE = 25,
        NDYM = 19
    };

    Eigen::Vector<double, 4> prev_motor_speed;
    Eigen::Vector<double, 4> motor_speed;
    Eigen::Vector<double, 3> goal_position;
    Eigen::Vector<double, 4> prev_action;

    Eigen::Vector<double, IDX::SIZE> x;

    Eigen::Vector<double, 3> pos() const { return x.segment<IDX::NPOS>(IDX::POS); }
    Eigen::Vector<double, 4> att() const { return x.segment<IDX::NATT>(IDX::ATT); }
    Eigen::Vector<double, 3> vel() const { return x.segment<IDX::NVEL>(IDX::VEL); }
    Eigen::Vector<double, 3> ome() const { return x.segment<IDX::NOME>(IDX::OME); }
    Eigen::Vector<double, 3> acc() const { return x.segment<IDX::NACC>(IDX::ACC); }
    Eigen::Vector<double, 3> tau() const { return x.segment<IDX::NTAU>(IDX::TAU); }
    Eigen::Vector<double, 3> bome() const { return x.segment<IDX::NBOME>(IDX::BOME); }
    void set_pos(const Eigen::Vector<double, 3> &pos) { x.segment<IDX::NPOS>(IDX::POS) = pos; }
    void set_att(const Eigen::Vector<double, 4> &att) { x.segment<IDX::NATT>(IDX::ATT) = att; }
    void set_vel(const Eigen::Vector<double, 3> &vel) { x.segment<IDX::NVEL>(IDX::VEL) = vel; }
    void set_ome(const Eigen::Vector<double, 3> &ome) { x.segment<IDX::NOME>(IDX::OME) = ome; }
    void set_acc(const Eigen::Vector<double, 3> &acc) { x.segment<IDX::NACC>(IDX::ACC) = acc; }
    void set_tau(const Eigen::Vector<double, 3> &tau) { x.segment<IDX::NTAU>(IDX::TAU) = tau; }
    void set_bome(const Eigen::Vector<double, 3> &bome) { x.segment<IDX::NBOME>(IDX::BOME) = bome; }

    /**
     * @brief Returns a quaternion from the state attitude vector.
     *
     * @return Quaternion
     */
    Eigen::Quaternion<double> q() const { return Eigen::Quaternion<double>(att()[0], att()[1], att()[2], att()[3]); }

    /**
     * @brief Set all values in the current state to zero and resets quadcopter position.
     *
     */
    void setZero()
    {
        x.setZero();
        x(IDX::ATTW) = 1.0;
        x(IDX::POSZ) = 0;
        x(IDX::ACCZ) = -9.81;
        prev_motor_speed.setZero();
        motor_speed.setZero();
        goal_position = Eigen::Vector<double, 3>{double(1), double(1), double(2)};
    }

    void setZero(bool doNotSetMotors)
    {
        x.setZero();
        x(IDX::ATTW) = 1.0;

        x(IDX::ATTW) = 1.0;
        x(IDX::POSZ) = 0;
    }

    /**
     * @brief Returns the current observation from the quadcopters state, ready for the policy.
     *
     * @return Vector<16> Observation vector
     */
    Eigen::Vector<double, 16> getObservation()
    {
        Eigen::Vector<double, 16> observation;
        Eigen::Vector<double, 3> distance_to_target = goal_position - pos();
        observation << distance_to_target, q().toRotationMatrix().eulerAngles(2, 1, 0), vel(), ome(), prev_action;
        return observation;
    }
};
/**
 * @brief Updates the accelerations of the quadcopter on the current state object
 *
 * @param state current state
 * @param motor_thrusts motor thrusts
 */
void update_accelerations(State &state, Eigen::Vector<double, 4> motor_thrusts)
{
    Eigen::Vector<double, 4> force_torques = B_allocation_ * motor_thrusts;
    Eigen::Vector<double, 3> force(0, 0, force_torques[0]);
    // linear acceleration
    state.set_acc(state.q() * force * 1.0 / mass + _gz);

    // angular acceleration
    Eigen::Vector<double, 3> torque = force_torques.tail(3);
    state.set_tau(torque);
}

/**
 * @brief Quarternion multiplication required for the update position function
 *
 * @param q
 * @return Eigen::Matrix<double, 4, 4>
 */
Eigen::Matrix<double, 4, 4>
Q_right(const Eigen::Quaternion<double> &q)
{
    return (Eigen::Matrix<double, 4, 4>() << q.w(), -q.x(), -q.y(), -q.z(), q.x(), q.w(), q.z(),
            -q.y(), q.y(), -q.z(), q.w(), q.x(), q.z(), q.y(), -q.x(), q.w())
        .finished();
}

/**
 * @brief Updates the position of the quadcopter, ideally using predefined acclerations and velocities
 *
 * @param state current state of the environment (including accelerations, velocities)
 * @param next_state next_state of the environment, with positions and rotations set
 * @return State The next state of the environment
 */
State update_position(State state, State next_state) // integrated dynamics
{
    next_state.set_pos(state.vel());
    next_state.set_att(0.5 * Q_right(Eigen::Quaternion<double>(0, state.ome()[0], state.ome()[1], state.ome()[2])) * state.att());
    next_state.set_vel(state.acc());
    next_state.set_ome((J_inv * (state.tau() - state.ome().cross(J * state.ome()))));
    return next_state;
}
const Eigen::Vector<double, 4> rk4_sum_vec = {1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0};

/**
 * @brief Runge-Kutta 4 integrator. Calls the update position functions and applies the integration
 *
 * @param initial initial starting state
 * @param dt Time step to integrate over
 * @param final State to be filled with updated state
 * @return State
 */
State integrate(State inital, double dt, State final)
{
    Eigen::Matrix<double, inital.x.rows(), 4> k = Eigen::Matrix<double, inital.x.rows(), 4>::Zero(inital.x.rows(), 4);

    final = inital;

    // K_1
    State k1;
    k1.setZero();
    k1 = update_position(final, k1);

    // K_2
    State k2;
    k2.setZero();
    final.x = inital.x + 0.5 * dt * k1.x;
    k2 = update_position(final, k2);

    // k_3
    State k3;
    k3.setZero();
    final.x = inital.x + 0.5 * dt * k2.x;
    k3 = update_position(final, k3);

    // k_4
    State k4;
    k4.setZero();
    final.x = inital.x + dt * k3.x;
    k4 = update_position(final, k4);

    k.col(0) = k1.x;
    k.col(1) = k2.x;
    k.col(2) = k3.x;
    k.col(3) = k4.x;

    final.x = inital.x + dt * k * rk4_sum_vec;

    return final;
}
/**
 * @brief Step motors function. This function is used to simulate the motor ramp up and ramp down
 *
 * @param init_rpm Initial motor speed
 * @param desired_rpm Desired motor Speed
 * @param dt Time with which to achieve motor speed
 * @return Eigen::Vector<double, 4> init_rpm, achieved rpm
 */

Eigen::Vector<double, 4> step_motors(Eigen::Vector<double, 4> init_rpm, Eigen::Vector<double, 4> desired_rpm, double dt)
{
    double c = std::exp(-dt * motor_tau_inv);
    init_rpm = c * init_rpm + (1 - c) * desired_rpm;
    return init_rpm;
}
/**
 * @brief Converts the thrusts from percentage to RPM. Note, this relationshsip might not be linear
 *
 * @param thrusts_pc Percentage of achieveable fthrust (from 0 to 1)
 * @return * Eigen::Vector<double, 4>  Conversion from thrust to RPM.
 */
Eigen::Vector<double, 4> thrust_to_rpm(Eigen::Vector<double, 4> thrusts_pc)
{
    return (thrusts_pc * 1000) + Eigen::Vector<double, 4>(100, 100, 100, 100);
}
State step(State state, Eigen::Vector<double, 4> &init_omega, Eigen::Vector<double, 4> desired_thrust_pc, double sim_dt)
{
    State old_state = state;
    State next_state = state;

    Eigen::Vector<double, 4> desired_rpm = thrust_to_rpm(desired_thrust_pc);

    double max_dt = 2.5e-3;
    double remain_ctrl_dt = sim_dt;
    Eigen::Vector<double, 4> motor_omega_ = init_omega;

    int i = 0;

    while (remain_ctrl_dt > 0)
    {
        Eigen::Vector<double, 4> recieved_rpm = state.prev_motor_speed;
        // delayed response to input command
        if (i >= 1)
        {
            recieved_rpm = desired_rpm;
        }
        i++;

        // time step
        double dt = std::min(remain_ctrl_dt, max_dt);
        remain_ctrl_dt -= dt;

        // clamp motor rpm
        Eigen::Vector<double, 4> clamped = recieved_rpm.cwiseMax(100.0f).cwiseMin(1100.0f);

        // rotor ramp up time
        motor_omega_ = step_motors(motor_omega_, clamped, dt);
        // clamp motor thrust
        Eigen::Vector<double, 4> motor_thrusts = motor_omega_.cwiseProduct(motor_omega_) * motor_constant;
        motor_thrusts = motor_thrusts.cwiseMax(0.0584f);
        motor_thrusts = motor_thrusts.cwiseMin(9.0f);
        // std::exit(0);

        // forces & moments
        update_accelerations(state, motor_thrusts);

        // integrate dynamics function
        next_state = integrate(state, dt, next_state);

        state = next_state;

        state.motor_speed = motor_omega_;
    }
    state.prev_motor_speed = desired_rpm;
    state.prev_action = desired_thrust_pc;

    return state;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double mse_pos(Eigen::Vector<double, 3> predicted_xyz, Eigen::Vector<double, 3> true_xyz)
{
    double se_x = pow(predicted_xyz[0] - true_xyz[0], 2);
    double se_y = pow(predicted_xyz[1] - true_xyz[1], 2);
    double se_z = pow(predicted_xyz[2] - true_xyz[2], 2);

    return (se_x + se_y + se_z); // mean error of each state variable. Weighting each variable equally
}

int main()
{
    State quad = State();
    quad.setZero();

    double lx = init_state[1];
    double ly = init_state[2];
    double lz = init_state[3];
    double roll = init_state[6];
    double pitch = init_state[5];
    double yaw = init_state[4];
    double vellx = init_state[7];
    double velly = init_state[8];
    double vellz = init_state[9];
    double angvelx = init_state[10];
    double angvely = init_state[11];
    double angvelz = init_state[12];
    double linaccx = init_state[13];
    double linaccy = init_state[14];
    double linaccz = init_state[15];
    double angaccx = init_state[16];
    double angaccy = init_state[17];
    double angaccz = init_state[18];
    double a0 = init_state[19];
    double a1 = init_state[20];
    double a2 = init_state[21];
    double a3 = init_state[22];
    quad.setZero();
    quad.set_pos(Eigen::Vector<double, 3>{lx, ly, lz});
    quad.goal_position = Eigen::Vector<double, 3>{1, 1, 2};
    Eigen::Quaternion<double> q;
    q = Eigen::AngleAxis<double>(yaw, Eigen::Vector<double, 3>::UnitZ()) * Eigen::AngleAxis<double>(pitch, Eigen::Vector<double, 3>::UnitY()) * Eigen::AngleAxis<double>(roll, Eigen::Vector<double, 3>::UnitX());
    quad.set_att(Eigen::Vector<double, 4>{q.w(), q.x(), q.y(), q.z()});
    quad.set_vel(Eigen::Vector<double, 3>{vellx, velly, vellz});
    quad.set_ome(Eigen::Vector<double, 3>{angvelx, angvely, angvelz});
    quad.set_acc(Eigen::Vector<double, 3>{linaccx, linaccy, linaccz});
    quad.set_tau(Eigen::Vector<double, 3>{angaccx, angaccy, angaccz});

    quad.prev_action = Eigen::Vector<double, 4>{hte, hte, hte, hte};
    quad.prev_motor_speed = thrust_to_rpm(quad.prev_action);

    auto start5 = std::chrono::high_resolution_clock::now();
    double total_se = 0;
    double count = 0;
    double prev_x = 0;
    int same_count = 0;
    NeuralNetwork nn = NeuralNetwork();

    for (int i = 0; i < 1000; i++)
    {

        Eigen::Vector<double, 4> actions = nn.forward_prop(quad.getObservation());

        // convert nn output to around hover point
        for (int j = 0; j < 4; j++)
        {
            if (actions[j] < 0)
            {
                actions[j] = (actions[j] * hte) + hte;
            }
            else
            {
                actions[j] = (actions[j] * remaining_thrust) + hte;
            }
        }

        quad = step(quad, quad.prev_motor_speed, actions, timestep[i + 1]);

        total_se += mse_pos(quad.pos(), Eigen::Vector<double, 3>{all_states[i][1], all_states[i][2], all_states[i][3]});

        // fail safe for failed simulation
        if (quad.getObservation()[0] == prev_x)
        {
            same_count++;
            if (same_count > 10)
            {
                std::cout << "False" << std::endl;
                return 0;
            }
        }
        else
        {
            prev_x = quad.getObservation()[0];
            same_count = 0;
        }
    }

    auto stop5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(stop5 - start5);

    // program only returns score
    std::cout << total_se / 3 << std::endl; // mse

    return 0;
}