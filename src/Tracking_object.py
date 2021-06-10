class Tracking_object(object):

    def __init__(self, class_name, tracking_id):
        self.class_name = class_name
        self.tracking_id = tracking_id
        self.state = 1
        self.last_directions = [0]*10
        self.otherDirectionCounter = 0
        self.sameDirectionCounter = 0
        self.stopCounter = 0
    
    def increase_state(self):
        state = self.state + 1
        self.state = state % 4
    def get_state(self):
        return self.state
    
    def get_last_directions(self):
        return self.last_directions
    def add_directions(self, update):
        self.last_directions.pop(0)
        self.last_directions.append(update)
        
        
    def get_other_direction_counter(self):
        return self.otherDirectionCounter
    
    def increase_same_direction_counter(self):
        self.sameDirectionCounter += 1
        if self.sameDirectionCounter == 10:
            if self.state == 0 or self.state == 2:
                self.increase_state()
                self.sameDirectionCounter = 0
                self.stopCounter = 0
            elif self.state == 1 or self.state == 3:
                self.sameDirectionCounter = 0
                self.stopCounter = 0

    def get_same_direction_counter(self):
        return self.sameDirectionCounter
    
    def increase_stop_counter(self):
        self.stopCounter += 1
        if self.stopCounter == 8:
            self.sameDirectionCounter = 0
            self.stopCounter = 0
            if self.state == 1 or self.state == 3:
                self.increase_state()
        
    def get_stop_counter(self):
        return self.stopCounter