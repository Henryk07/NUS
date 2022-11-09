#Haolin Chen#
#A0260816E#
#ESP3201#
import numpy as np
import scipy.stats
from numpy.random import uniform, randn, random


def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits
    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles


def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound
    # (jump from -pi to pi). Therefore, we generate unit vectors from the
    # angles and calculate the angle of their average

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        # make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    # calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]


def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.15, 0.15]

    # standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    sigma_delta_trans = noise[2] * delta_trans + \
        noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans

    # generate new particle set after motion update
    new_particles = []

    for particle in particles:
        new_particle = dict()
        # sample noisy motions
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + \
            np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

        # calculate new particle pose
        new_particle['x'] = particle['x'] + \
            noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        new_particle['y'] = particle['y'] + \
            noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        new_particle['theta'] = particle['theta'] + \
            noisy_delta_rot1 + noisy_delta_rot2
        new_particles.append(new_particle)
    return new_particles


def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []
    # weights = np.array([])
    '''your code here'''
    # get the position of the landmark
    N = len(particles)
    NL = len(landmarks)
    particles_tempx = [0] * (N)
    particles_tempy = [0] * (N)
    # convert the particles' format and types
    for i in range(N):
        particles_tempx[i] = particles[i]['x']
    array_particles = np.zeros((N, 2))
    array_t = np.array(particles_tempx)
    for i in range(N):
        array_particles[i, 0] = array_t[i]

    for i in range(N):
        particles_tempy[i] = particles[i]['y']

    array_t = np.array(particles_tempy)
    for i in range(N):
        array_particles[i, 1] = array_t[i]
    # print('array', array_particles[:, :], type(array_particles))
    # calc and compare the distances
    p_range = np.zeros((N, NL))
    for i in range(N):
        px = array_particles[i][0]
        py = array_particles[i][1]
        # get particles x and y
        # print(px, py)
        # j = 0
        for lm_id in landmarks.keys():

            lx = landmarks[lm_id][0]
            ly = landmarks[lm_id][1]

        # get landmark x and y
            p_range[i][lm_id-1] = np.sqrt((lx - px)**2 + (ly - py)**2) + \
                np.random.normal(loc=0.0, scale=sigma_r)

            # j = j+1

    weights = np.zeros((N, 1))
    error = np.zeros((N, 1))
    for i in range(N):
        total_buffer = 0
        for j in range(NL):
            buffer = 0
            buffer = (p_range[i][j-1]-ranges[j-1])**2
            total_buffer += buffer

        error[i] = np.sqrt(total_buffer) + \
            abs(np.random.normal(loc=0.0, scale=sigma_r))
    # reverse the weights, since lower error means closer
        weights[i] = 1/error[i]
        #total_wei += weights[i]

        # print(weights[i])
    # print(b=[x[0] for x in particles_temp])
    weights += 1.e-300  # avoid round-off to zero

    weights /= sum(weights)  # normalize

    '''***        ***'''

    # normalize weights
    # normalizer = sum(weights)
    # weights = weights[0] / normalizer

    return weights


def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []

    # print('re', weights[10])
    N = len(particles)
    probabilites = weights / np.sum(weights)
    probabilites_new = np.reshape(
        probabilites, (np.product(probabilites.shape),))
    #probabilites_new = np.squeeze(weights)
    new_index = np.random.choice(N, size=100, p=probabilites_new)
    new_particles = np.array(particles)[new_index]

    '''
    
    
    '''

    return new_particles
