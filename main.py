from constants import CONSTANTS as C
from autonomous_vehicle import AutonomousVehicle
from sim_draw import Sim_Draw
from sim_data import Sim_Data
import pickle
import os
import pygame as pg
import datetime
import timeit


#import resource
import loss_functions

class Main():

    def __init__(self):

        # Setup
        self.duration =100
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
                                       inference_type="non empathetic")  #M
        self.car_2 = AutonomousVehicle(scenario_parameters=self.P,
                                       car_parameters_self=self.P.CAR_2,
                                       loss_style="reactive",
                                       who=0,
                                       inference_type="non empathetic")  #H

        # Assign 'other' cars
        self.car_1.other_car = self.car_2
        self.car_2.other_car = self.car_1
        self.car_1.states_o = self.car_2.states
        self.car_2.states_o = self.car_1.states
        self.car_1.actions_set_o = self.car_2.actions_set
        self.car_2.actions_set_o = self.car_1.actions_set
        self.car_1.does_inference= True
        self.car_2.does_inference= True

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
        self.show_trajectory = True
        self.show_loss = True
        self.show_predicted_states_others = True
        self.show_does_inference = True

        self.trial()

    def trial(self):
        frequency = C.FREQUENCY # frequency every 1/3rd of the time
        counter = 0

        #timing the processing time and resource
        start_timer = timeit.default_timer()


        while self.running:

            # Update model here
            if not self.paused:
                # if counter% frequency == 0 or counter < 1: # skipping first three
                #     skip_update = False
                #     print("counter val =", counter)
                #
                # else:
                #     skip_update = True
                # data_count = counter% frequency


                #threhold based calculation
                #car_2.states(...)- car_1.predictedothers() > distance
                #car 1 threshold
                skip_update_car1 = False
                skip_update_car2 = False
                # if counter > -1:
                #     # skip_update_car1 = True
                #     # skip_update_car2 = True
                #     if len(self.car_1.predicted_actions_other) < 2:
                #         if np.abs(self.car_1.predicted_actions_other[0][0][1] - self.car_2.states[-1][1]) < 0.001:
                #             skip_update_car1 = True
                #             print(skip_update_car1)

                    #car 2 threshold
                    # if len(self.car_2.predicted_actions_other) < 2:
                    #     if np.abs(self.car_2.predicted_actions_other[0][0][0] - self.car_1.states[-1][0])< 0.001:
                    #         skip_update_car2 = True
                    #         print(skip_update_car2)

                self.car_1.update(self.frame, skip_update_car1)
                self.car_2.update(self.frame, skip_update_car2)
                counter+= 1
                print("counter" , counter )

                # calculate gracefulness old
                # grace = []
                # for wanted_trajectory_other in self.car_2.wanted_trajectory_other:
                #     wanted_actions_other = self.car_2.dynamic(wanted_trajectory_other)
                #     grace.append(1000*(self.car_1.states[-1][0] - wanted_actions_other[0][0]) ** 2)
                # self.car_1.social_gracefulness.append(sum(grace*self.car_2.inference_probability))

                grace = []
                for wanted_trajectory_other in self.car_2.wanted_trajectory_other:
                    wanted_actions_other = self.car_2.dynamic(wanted_trajectory_other)
                    grace.append(1000*(self.car_1.states[-1][0] - wanted_actions_other[0][0]) ** 2)

                if not len(self.car_2.wanted_inference_probability) == 0:
                    self.car_1.social_gracefulness.append(sum(grace*self.car_2.wanted_inference_probability))


                #self.wanted_inference_probability


                #calculate instant loss
                import numpy as np
                intent_loss_car_1 = self.car_1.intent * np.exp(C.EXPTHETA * (- self.car_1.temp_action_set[C.ACTION_TIMESTEPS-1][0] + 0.6))
                intent_loss_car_2 = self.car_2.intent * np.exp(C.EXPTHETA * (self.car_2.temp_action_set[C.ACTION_TIMESTEPS-1][1] + 0.6))
                #D =  np.sqrt(np.sum((np.array(self.car_1.states) - np.array(self.car_2.states))**2)) + 1e-12 # np.sum((my_pos - other_pos)**2, axis=1)
                #D = np.sqrt(self.car_1.states[-1][0] * self.car_1.states[-1][0] + self.car_2.states[-1][1] * self.car_2.states[-1][1])
                predicted_distance = np.sum((self.car_1.temp_action_set - self.car_2.temp_action_set) ** 2, axis=1)
                D = np.sqrt(predicted_distance)
                collision_loss = np.sum(np.exp(C.EXPCOLLISION * (-D + C.CAR_LENGTH ** 2 * 1.5)))
                #collision_loss =
                print(intent_loss_car_1)
                print(collision_loss)
                print(intent_loss_car_2)
                plannedloss_car1 = intent_loss_car_1+ collision_loss
                plannedloss_car2 = intent_loss_car_2+ collision_loss

                # plannedloss_car1 = 0
                # plannedloss_car2 = 0

                #print(plannedloss_car1)
                # print("trajectory other", self.car_1.predicted_trajectory_other)
                # print("inference probability", self.car_1.inference_probability)
                #
                # print("predicted trajectory of self", self.car_1.wanted_trajectory_self)




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
                                          social_gracefulness=self.car_1.social_gracefulness,
                                          planned_loss=plannedloss_car1,
                                          does_inf=not(skip_update_car1),
                                          predicted_trajectory_other=self.car_1.predicted_trajectory_set_other,
                                          collision_loss=collision_loss)

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
                                          theta_probability=self.car_2.theta_probability,
                                          planned_loss=plannedloss_car2,
                                          does_inf = not(skip_update_car2),
                                          predicted_trajectory_other= self.car_2.predicted_trajectory_set_other)

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
        # memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0 #maximum residential set size
        # print(memMb)

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
            # for i in range(1,self.frame+1):
            #     if car_2_theta[i,0] == 1:
            #         print(i)
            #         break
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

        if self.show_action:
            car_1_action = np.empty((0, 2))
            car_2_action = np.empty((0, 2))
            car_1_action_predicted = np.empty((0, 2))
            for t in range(self.frame):
                car_1_action = np.append(car_1_action, (np.expand_dims(self.sim_data.car1_actions[t+1], axis=0) - np.expand_dims(self.sim_data.car1_actions[t], axis=0)), axis=0)
                car_2_action = np.append(car_2_action, (np.expand_dims(self.sim_data.car2_actions[t+1], axis=0) - np.expand_dims(self.sim_data.car2_actions[t], axis=0)), axis=0)
                # car_1_action = np.append(car_1_action, np.expand_dims(self.sim_data.car1_actions[t], axis=0), axis=0)
                # car_2_action = np.append(car_2_action, np.expand_dims(self.sim_data.car2_actions[t], axis=0), axis=0)
                # car_1_action_predicted = np.append(car_1_action_predicted, np.expand_dims(self.sim_data.car1_predicted_others_prediction_of_my_actions[t], axis=0), axis=0 )
                # car_2_action_predicted = np.append(car_2_action_predicted, np.expand_dims(self.sim_data.car1_predicted_actions_other[t], axis=0), axis=0 )
            #dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])

            # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
            # plt.legend()

            fig1, (ax1, ax2) = plt.subplots(2) #3 rows
            # fig1.suptitle('Euclidean distance and Agent States')
            # ax1.plot(dist, label='car dist')
            # ax1.legend()
            # ax1.set(xlabel='time', ylabel='distance')

            ax1.plot(range(1,self.frame+1), car_1_action[:,0], label='car 1 actual action')
            #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='action')

            ax2.plot(range(1,self.frame+1), car_2_action[:,0], label='car 2 actual action')
            #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='action')
            #plt.show()

        if self.show_loss:
            car_1_loss = np.empty((0, 1))
            car_2_loss = np.empty((0, 1))
            car_collision_loss = np.empty((0, 1))

            for t in range(self.frame):
                #def calculate_instanteous_reactive_loss(self, theta_self, trajectory, trajectory_other, s_self, s_other,s, probability):
                #car_1_instant = calculate_instanteous_reactive_loss(self.sim_data.car_1_theta, self.sim_data.car1_trajectory[t])
                # car_2_instant = calculate_instanteous_reactive_loss()
                car_1_loss = np.append(car_1_loss, self.sim_data.car1_planned_loss[t])
                car_2_loss = np.append(car_2_loss, self.sim_data.car2_planned_loss[t])
                car_collision_loss = np.append(car_collision_loss, self.sim_data.car1_collision_loss[t])

                # car_1_action_predicted = np.append(car_1_action_predicted, np.expand_dims(self.sim_data.car1_predicted_others_prediction_of_my_actions[t], axis=0), axis=0 )
                # car_2_action_predicted = np.append(car_2_action_predicted, np.expand_dims(self.sim_data.car1_predicted_actions_other[t], axis=0), axis=0 )
            #dist = np.sqrt(car_1_state[:,0] *car_1_state[:,0] + car_2_state[:,1] * car_2_state[:,1])

            # plt.plot(range(1,self.frame+1), car_1_state[:,0], label='car 1 M')
            # plt.plot(range(1,self.frame+1), car_2_state[:,1], label='car 2 H', linestyle='--')
            # plt.legend()

            fig1, (ax1, ax2, ax3) = plt.subplots(3) #3 rows
            # fig1.suptitle('Euclidean distance and Agent States')
            # ax1.plot(dist, label='car dist')
            # ax1.legend()
            # ax1.set(xlabel='time', ylabel='distance')

            ax1.plot(range(1,self.frame+1), car_1_loss, label='car 1 loss')
            #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='instant loss')

            ax2.plot(range(1,self.frame+1), car_2_loss, label='car 2 loss')
            #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='instant loss')

            ax3.plot(range(1, self.frame + 1), car_collision_loss, label='collision loss')
            # ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax3.legend()
            ax3.set(xlabel='time', ylabel='collision loss over next 100 timesteps')
            plt.show()

        if self.show_does_inference:
            car_1_does_inference = np.empty((0, 1))
            car_2_does_inference = np.empty((0, 1))

            for t in range(self.frame):
                car_1_does_inference = np.append(car_1_does_inference, self.sim_data.car1_does_inference[t])
                car_2_does_inference = np.append(car_2_does_inference, self.sim_data.car2_does_inference[t])


            fig1, (ax1, ax2) = plt.subplots(2) #3 rows
            ax1.plot(range(1,self.frame+1), car_1_does_inference, label='car 1 inference')
            #ax1.plot(range(1,self.frame+1), car_1_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='does inference')

            ax2.plot(range(1,self.frame+1), car_2_does_inference, label='car 2 inference')
            #ax2.plot(range(1,self.frame+1), car_2_action_predicted[:,0], label='car 1 prediction of car 2 prediction of car 1')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='does inference')
            plt.show()

        if self.show_trajectory:
            car_1_predicted_trajectory_1 = np.empty((0))
            car_1_predicted_trajectory_2 = np.empty((0))
            car_2_predicted_trajectory_1 = np.empty((0))
            car_2_predicted_trajectory_2 = np.empty((0))
            car_1_planned_trajectory = np.empty((0))
            car_2_planned_trajectory = np.empty((0))
            car_1_timestep_2 = np.empty((0))
            car_2_timestep_2 = np.empty((0))
            # # car_1_performed_trajectory = np.empty((0, 1))
            # # car_2_performed_trajectory = np.empth((0, 1))
            #
            for t in range(self.frame):
                car_1_predicted_trajectory_1 = np.append(car_1_predicted_trajectory_1, self.sim_data.car2_predicted_trajectory_other[0][t][0][0])
                car_1_planned_trajectory = np.append(car_1_planned_trajectory, self.sim_data.car1_planned_trajectory_set[t][0])
                if len(self.sim_data.car2_predicted_trajectory_other[0][t]) == 2:
                    car_1_predicted_trajectory_2 = np.append(car_1_predicted_trajectory_2,
                                                             self.sim_data.car2_predicted_trajectory_other[0][t][1][0])
                    car_1_timestep_2 = np.append(car_1_timestep_2, t)

                car_2_predicted_trajectory_1 = np.append(car_2_predicted_trajectory_1, self.sim_data.car1_predicted_trajectory_other[0][t][0][0])
                car_2_planned_trajectory = np.append(car_2_planned_trajectory,
                                                     self.sim_data.car2_planned_trajectory_set[t][0])
                if len(self.sim_data.car1_predicted_trajectory_other[0][t]) == 2:
                    car_2_predicted_trajectory_2 = np.append(car_2_predicted_trajectory_2,
                                                             self.sim_data.car1_predicted_trajectory_other[0][t][1][0])
                    car_2_timestep_2 = np.append(car_2_timestep_2, t)
                #car_1_performed_trajectory = np.append(car_1_performed_trajectory, self.sim_data.car1_planned_)

            fig1, (ax1, ax2) = plt.subplots(2)
            # ax1.plot(range(1,self.frame+1), self.sim_data.car2_predicted_trajectory_other[0][1: self.frame+1], label='predicted trajectory of car 1(1)')
            # ax1.plot(range(1, self.frame + 1), self.sim_data.car2_predicted_trajectory_other[1] [1: self.frame+1],
            #          label='predicted trajectory of car 1(2)')
            # ax1.plot(range(1,self.frame+1), self.sim_data.car1_planned_trajectory_set, label='actual trajectory of car 1')
            # ax1.legend()
            # ax1.set(xlabel='time', ylabel='trajectory')

            ax1.plot(range(0,self.frame), car_1_predicted_trajectory_1, label='predicted trajectory of car 1(1)')
            ax1.plot(car_1_timestep_2 ,car_1_predicted_trajectory_2,
                     label='predicted trajectory of car 1(2)')
            ax1.plot(range(0,self.frame), car_1_planned_trajectory, label='actual trajectory of car 1')
            ax1.legend()
            ax1.set(xlabel='time', ylabel='trajectory')

            ax2.plot(range(0,self.frame), car_2_predicted_trajectory_1, label='predicted trajectory of car 2 (1)')
            ax2.plot(car_2_timestep_2, car_2_predicted_trajectory_2, label='predicted trajectory of car 2 (2)')
            ax2.plot(range(0,self.frame), car_2_planned_trajectory, label='actual trajectory of car 2')
            ax2.legend()
            ax2.set(xlabel='time', ylabel='trajectory')
            plt.show()



if __name__ == "__main__":
    Main()