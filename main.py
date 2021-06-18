from constants import CONSTANTS as C
from autonomous_vehicle import AutonomousVehicle
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import os
import pygame as pg
import datetime
import timeit

class Main():

    def __init__(self):

        # Setup
        self.duration = 100
        self.P = C.PARAMETERSET_2  # Scenario parameters choice
        # Time handling
        self.clock = pg.time.Clock()
        self.fps = C.FPS
        self.running = True
        self.paused = False
        self.end = False
        self.frame = 0
        self.car_num_display = 0

        # Sim output
        self.sim_data = Sim_Data()

        # self.sim_out = open("./sim_outputs/output_test.pkl", "wb")c

        # Vehicle Definitions ('aggressive','reactive','passive_aggressive',"berkeley_courtesy", 'courteous')
        self.car_1 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_1,
                                       loss_style="reactive",
                                       who=1,
                                       inference_type="empathetic")  #M
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       loss_style="reactive",
                                       who=0,
                                       inference_type="empathetic")  #H

        # Assign 'other' cars
        self.car_1.other_car = self.car_2
        self.car_2.other_car = self.car_1
        self.car_1.states_o = self.car_2.states
        self.car_2.states_o = self.car_1.states
        self.car_1.actions_set_o = self.car_2.actions_set
        self.car_2.actions_set_o = self.car_1.actions_set

        if C.DRAW:
            self.sim_draw = Sim_Draw(self.P, C.ASSET_LOCATION)
            pg.display.flip()
            # self.capture = True if input("Capture video (y/n): ") else False
            self.capture = False
            self.output_data_pickle = False
            output_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            if self.output_data_pickle or self.capture:
                os.makedirs("./sim_outputs/%s" % output_name)
            if self.output_data_pickle:
                self.sim_out = open("./sim_outputs/%s/output.pkl" % output_name, "wb")

            if self.capture:
                self.output_dir = "./sim_outputs/%s/video/" % output_name
                os.makedirs(self.output_dir)

        # Go

        #figures printing and saving video
        self.show_prob_theta = True
        self.show_states = True
        self.show_action = True

        self.trial()

    def trial(self):
        frequency = 3
        counter = 0

        #timing the processing time
        start_timer = timeit.default_timer()

        while self.running:

            # Update model here
            if not self.paused:
                if counter% frequency == 0 or counter <= 2: # skipping first three
                    skip_update = False

                else:
                    skip_update = True
                self.car_1.update(self.frame, skip_update)
                self.car_2.update(self.frame, skip_update)
                counter+= 1

                # calculate gracefulness
                grace = []
                for wanted_trajectory_other in self.car_2.wanted_trajectory_other:
                    wanted_actions_other = self.car_2.dynamic(wanted_trajectory_other)
                    grace.append(1000*(self.car_1.states[-1][0] - wanted_actions_other[0][0]) ** 2)
                self.car_1.social_gracefulness.append(sum(grace*self.car_2.inference_probability))

                # Update data
                self.sim_data.append_car1(states=self.car_1.states,
                                          actions=self.car_1.actions_set,
                                          action_sets=self.car_1.planned_actions_set,
                                          trajectory = self.car_1.planned_trajectory_set,
                                          predicted_theta_other=self.car_1.predicted_theta_other,
                                          predicted_theta_self=self.car_1.predicted_theta_self,
                                          predicted_actions_other=self.car_1.predicted_actions_other,
                                          predicted_others_prediction_of_my_actions=
                                          self.car_1.predicted_others_prediction_of_my_actions,
                                          wanted_trajectory_self=self.car_1.wanted_trajectory_self,
                                          wanted_trajectory_other=self.car_1.wanted_trajectory_other,
                                          wanted_states_other=self.car_1.wanted_states_other,
                                          inference_probability=self.car_1.inference_probability,
                                          inference_probability_proactive=self.car_1.inference_probability_proactive,
                                          theta_probability=self.car_1.theta_probability,
                                          social_gracefulness=self.car_1.social_gracefulness)

                self.sim_data.append_car2(states=self.car_2.states,
                                          actions=self.car_2.actions_set,
                                          action_sets=self.car_2.planned_actions_set,
                                          trajectory=self.car_2.planned_trajectory_set,
                                          predicted_theta_other=self.car_2.predicted_theta_other,
                                          predicted_theta_self=self.car_2.predicted_theta_self,
                                          predicted_actions_other=self.car_2.predicted_actions_other,
                                          predicted_others_prediction_of_my_actions=
                                          self.car_2.predicted_others_prediction_of_my_actions,
                                          wanted_trajectory_self=self.car_2.wanted_trajectory_self,
                                          wanted_trajectory_other=self.car_2.wanted_trajectory_other,
                                          wanted_states_other=self.car_2.wanted_states_other,
                                          inference_probability=self.car_2.inference_probability,
                                          inference_probability_proactive=self.car_2.inference_probability_proactive,
                                          theta_probability=self.car_2.theta_probability,)

            if self.frame >= self.duration:
                break

            if C.DRAW:
                # Draw frame
                self.sim_draw.draw_frame(self.sim_data, self.car_num_display, self.frame)

                if self.capture:
                    pg.image.save(self.sim_draw.screen, "%simg%03d.jpeg" % (self.output_dir, self.frame))

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        self.running = False

                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_p:
                            self.paused = not self.paused

                        if event.key == pg.K_q:
                            pg.quit()
                            self.running = False

                        if event.key == pg.K_d:
                            self.car_num_display = ~self.car_num_display

                # Keep fps
                # self.clock.tick(self.fps)

            if not self.paused:
                self.frame += 1

        pg.quit()
        stop_timer = timeit.default_timer()
        print('Time: ', stop_timer - start_timer)

        if self.output_data_pickle:
            pickle.dump(self.sim_data, self.sim_out, pickle.HIGHEST_PROTOCOL)
            print('Output pickled and dumped.')
        if self.capture:
            # Compile to video
            # os.system("ffmpeg -f image2 -framerate 1 -i %simg%%03d.jpeg %s/output_video.mp4 " % (self.output_dir, self.output_dir))
            img_list = [self.output_dir+"img"+str(i).zfill(3)+".jpeg" for i in range(self.frame)]
            import imageio
            images = []
            for filename in img_list:
                images.append(imageio.imread(filename))
            imageio.mimsave(self.output_dir+'movie.gif', images)
            #
            # # Delete images
            # [os.remove(self.output_dir + file) for file in os.listdir(self.output_dir) if ".jpeg" in file]
            # print("Simulation video output saved to %s." % self.output_dir)
        print("Simulation ended.")

        import matplotlib.pyplot as plt
        import numpy as np
        if self.show_prob_theta:
            car_1_theta = np.empty((0, 2))
            car_2_theta = np.empty((0, 2))
            for t in range(self.frame):
                car_1_theta = np.append(car_1_theta, np.expand_dims(self.sim_data.car2_theta_probability[t], axis=0), axis=0)
                car_2_theta = np.append(car_2_theta, np.expand_dims(self.sim_data.car1_theta_probability[t], axis=0), axis=0)

            plt.subplot(2, 1, 1)
            plt.title("Probability graph of the vehicle")
            plt.plot(range(1,self.frame+1), car_1_theta[:,0], label = "$\hat{\Theta}_M$= 1" )
            plt.plot(range(1,self.frame+1), car_1_theta[:,1], label = "$\hat{\Theta}_M$= 10^3")
            plt.ylabel("$p(\hat{\Theta}_M)$")
            plt.xlabel("frame")
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(range(1,self.frame+1), car_2_theta[:,0], label = "$\hat{\Theta}_H$= 1" )
            plt.plot(range(1,self.frame+1), car_2_theta[:,1], label = "$\hat{\Theta}_H$= 10^3" )
            plt.ylabel("$p(\hat{\Theta}_H)$")
            plt.xlabel("frame")
            plt.legend()

            plt.show()
            #plt.savefig('saved_figure.png')
        if self.show_states:
            car_1_state = np.empty((0, 2))
            car_2_state = np.empty((0, 2))
            for t in range(self.frame):
                car_1_state = np.append(car_1_state, np.expand_dims(self.sim_data.car1_states[t], axis=0), axis=0)
                car_2_state = np.append(car_2_state, np.expand_dims(self.sim_data.car2_states[t], axis=0), axis=0)
            dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])

            # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
            # plt.legend()

            fig1, (ax1, ax2, ax3) = plt.subplots(3) #3 rows
            fig1.suptitle('Euclidean distance and Agent States')
            ax1.plot(dist, label='car dist')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='distance')

            ax2.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='states')

            ax3.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H')
            ax3.legend()
            ax3.set(xlabel='time', ylabel='states')
            plt.show()



if __name__ == "__main__":
    Main()