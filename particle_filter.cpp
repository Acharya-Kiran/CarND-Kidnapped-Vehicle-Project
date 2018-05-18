/*
 * particle_filter.cpp
 *
 *  Created on: April 12, 2018
 *      Author: Kiran Acharya
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>


#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 50;
	
	normal_distribution<double> dist_x(0.0, std[0]);
	normal_distribution<double> dist_y(0.0, std[1]);
	normal_distribution<double> dist_theta(0.0, std[2]);
	
	for(unsigned int i=0;i<num_particles;i++){
		Particle P;
		P.id = i;
		P.x = x + dist_x(gen);
		P.y = y + dist_y(gen);
		P.theta = theta + dist_theta(gen);
		P.weight = 1.0;
		particles.push_back(P);
	}
	
	is_initialized = true;
		
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	
	for(unsigned int i=0;i<num_particles;i++){
		if(fabs(yaw_rate)<0.00001){
			particles[i].x += velocity*cos(particles[i].theta)*delta_t;
			particles[i].y += velocity*sin(particles[i].theta)*delta_t;
		}
		else{
			particles[i].x += (velocity*1.0/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta) );
			particles[i].y += (velocity*1.0/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta += yaw_rate*delta_t;
		}
		
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
		

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i=0;i<observations.size();i++){
		
		LandmarkObs obs = observations[i];
		double minimum_distance = 99.9e+100;
		int id_ = -1;
		
		for(int j=0;j<predicted.size();j++){
			LandmarkObs pred = predicted[j];
			double dist_ = sqrt((obs.x-pred.x)*(obs.x-pred.x)+(obs.y-pred.y)*(obs.y-pred.y));
			if(dist_ < minimum_distance){
				minimum_distance = dist_;
				id_ = pred.id;
				
			}
				
		}
		observations[i].id = id_;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	
	for(unsigned int i=0;i<particles.size();i++){	
        vector<LandmarkObs> filtered_landmarks;	
		for(unsigned int j=0;j<map_landmarks.landmark_list.size();j++){
			if(fabs(map_landmarks.landmark_list[j].x_f-particles[i].x)<=sensor_range && fabs(map_landmarks.landmark_list[j].y_f-particles[i].y)<=sensor_range ){
				filtered_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f});
			}
		}
		vector<LandmarkObs> transformed_observations;
		for(unsigned int j=0;j<observations.size();j++){
			double tx = particles[i].x + cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y;
			double ty = particles[i].y + sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y;
			transformed_observations.push_back(LandmarkObs{observations[j].id,tx,ty});
		}
		
		dataAssociation(filtered_landmarks,transformed_observations);
		particles[i].weight = 1.0;
		float mu_x,mu_y;
		for(unsigned int p=0;p<transformed_observations.size();p++){
			for(unsigned int q=0;q<filtered_landmarks.size();q++){
				if(filtered_landmarks[q].id == transformed_observations[p].id ){
					mu_x = filtered_landmarks[q].x;
					mu_y = filtered_landmarks[q].y;
					
				}
				
			}
			double gauss_norm = (1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]));
			double exponent = pow(transformed_observations[p].x - mu_x,2)/(2.0 * std_landmark[0]*std_landmark[0]) + pow(transformed_observations[p].y - mu_y,2)/(2.0 *std_landmark[1]*std_landmark[1]);
			double weight_= gauss_norm * exp( -1.0*exponent);
			//cout<<weight_<<endl;
			particles[i].weight *= weight_;

		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<double> weights;
	for(int i=0;i<particles.size();i++){
		weights.push_back(particles[i].weight);
	}
	vector<Particle> new_particles;
	double beta = 0.0;
	uniform_int_distribution<int> dist_index(0,particles.size()-1);
	unsigned int index = dist_index(gen);
	double max_weight = *max_element(weights.begin(),weights.end());
	uniform_real_distribution<double> dist_weights(0,2*max_weight);
	for(int j=0;j<particles.size();j++){
		beta = beta + dist_weights(gen);
		while(weights[index]<beta){
			beta = beta - weights[index];
			index = (index + 1)%num_particles;
		}	
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
