import matplotlib.pyplot as plt
from math import sqrt
 
class SwissRoll():
    def __init__(self):
        self.N = 0
             
    def gen_locations(self, N):
        locations = {}
        for i in range(N):
            for j in range(N):
                #locations.append((i,j))
                locations[(i,j)] = 0
                #print(locations)
        return locations
     
    def sort_locations(self, locs):
        N = int(sqrt(len(locs)))
        sorted_locs = {}
        sorted_locs[1] = (0,N-1)
        locs[sorted_locs[1]] = 1
        direction = 0
        def update_direction(loc, current_direction):
            if current_direction == 0:
                loc_next = (loc[0]+1,loc[1])
                if loc_next not in locs or locs[loc_next] == 1:
                    return 1
                else:
                    return 0
            elif current_direction == 1:
                loc_next = (loc[0],loc[1]-1)
                if loc_next not in locs or locs[loc_next] == 1:
                    return 2
                else:
                    return 1            
            elif current_direction == 2:
                loc_next = (loc[0]-1,loc[1])
                if loc_next not in locs or locs[loc_next] == 1:
                    return 3
                else:
                    return 2  
            elif current_direction == 3:
                loc_next = (loc[0],loc[1]+1)
                if loc_next not in locs or locs[loc_next] == 1:
                    return 0
                else:
                    return 3
         
        def next_loc(loc, direction):
            if direction == 0:
                return (loc[0]+1,loc[1])
            elif direction == 1:
                return (loc[0],loc[1]-1)
            elif direction == 2:
                return (loc[0]-1,loc[1])
            elif direction == 3:
                return (loc[0],loc[1]+1)
                 
        for i in range(2, N*N+1):
            direction = update_direction(sorted_locs[i-1], direction)
            loc_next = next_loc(sorted_locs[i-1], direction)
            sorted_locs[i] = loc_next
            locs[loc_next] = 1
            #print(loc_next)
         
        #print(locs)
        return sorted_locs
    def set_N(self):
        self.N = input("Please input number:\n")
         
    def show(self):
        locations = self.gen_locations(self.N)
        sorted_locations = self.sort_locations(locations)
        #print(locations)
        #print(sorted_locations)
         
        fig, ax = plt.subplots()
        for index in sorted_locations:
            plt.text(sorted_locations[index][0]+0.5,sorted_locations[index][1]+0.5,str(index),color='k')
         
        plt.axis((0,self.N,0,self.N))
        plt.xticks(range(self.N))
        plt.yticks(range(self.N))
        plt.grid(color='k')
        ax.tick_params(labelbottom=False, labelleft=False, length=0) 
        ax.set_aspect(1.0)
        plt.show()
         
if __name__ == "__main__":
    sw = SwissRoll()
    sw.set_N()
    sw.show()