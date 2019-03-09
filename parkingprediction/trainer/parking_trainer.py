from parkingprediction.trainer.base_trainer import BaseTrain

class ParkingTrainer(BaseTrain):

    def __init__(self, session, model, dataset, config):
        super(ParkingTrainer, self).__init__(session, model, dataset, config)

    def train_epoch(self, cur_epoche):

        num_iterations = self.config.num_iterations()

        mean_loss = 0
        mean_acc = 0
        for i in range(num_iterations):

            loss, acc = self.train_step()

            mean_loss += loss
            mean_acc += acc

        mean_loss /= num_iterations
        mean_acc /= num_iterations

        return mean_loss, mean_acc


    def train_step(self):

        _, loss, acc = self.session.run([self.model.opt, self.model.loss, self.model.acc], feed_dict={
                            self.iterator.handle_placeholder: self.train_handle
                      })

        self.session.run(self.model.increment_global_step_tensor)

        return loss, acc

    def test_step(self):

        loss, acc = self.session.run([self.model.loss, self.model.acc], feed_dict={
                            self.iterator.handle_placeholder: self.test_handle
                      })

        return loss, acc


    def log_progress(self, input, num_iteration, mode):

        summaries_dict = {
            'loss': input[0],
            'acc': input[1],
        }

        self.logger.log_scalars(num_iteration, summarizer=mode, summaries_dict=summaries_dict)

    def update_progress_bar(self, t_bar, train_output, test_output):

        t_bar.set_postfix(
            train_loss='{:05.3f}'.format(train_output[0]),
            train_acc='{:05.3f}'.format(train_output[1]),
            test_loss='{:05.3f}'.format(test_output[0]),
            test_acc='{:05.3f}'.format(test_output[1]),
        )
