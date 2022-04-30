"""
Based on the Jos Stam paper https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games
and the mike ash vulgarization https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://github.com/Guilouf/python_realtime_fluidsim
"""
import numpy as np
import math
import sys, argparse, os

 
ponerVal=[] 
ingreso=0
valores=[]
colores=''
valorF=0
objPx=0
objPy=0
objQ=0
objP=[]
object = False




class Fluid:

    def __init__(self):
        self.rotx = 1
        self.roty = 1
        self.cntx = 1
        self.cnty = -1

        self.size = 60  # map size
        self.dt = 0.2  # time interval
        self.iter = 2  # linear equation solving iteration number

        self.diff = 0.0000  # Diffusion
        self.visc = 0.0000  # viscosity

        self.s = np.full((self.size, self.size), 0, dtype=float)        # Previous density
        self.density = np.full((self.size, self.size), 0, dtype=float)  # Current density

        # array of 2d vectors, [x, y]
        self.velo = np.full((self.size, self.size, 2), 0, dtype=float)
        self.velo0 = np.full((self.size, self.size, 2), 0, dtype=float)

    def step(self):
        self.diffuse(self.velo0, self.velo, self.visc)

        # x0, y0, x, y
        #The trust? from my vectors to the neigbors
        self.project(self.velo0[:, :, 0], self.velo0[:, :, 1], self.velo[:, :, 0], self.velo[:, :, 1])

        self.advect(self.velo[:, :, 0], self.velo0[:, :, 0], self.velo0)
        self.advect(self.velo[:, :, 1], self.velo0[:, :, 1], self.velo0)

        self.project(self.velo[:, :, 0], self.velo[:, :, 1], self.velo0[:, :, 0], self.velo0[:, :, 1])

        self.diffuse(self.s, self.density, self.diff)

        self.advect(self.density, self.s, self.velo)

    def lin_solve(self, x, x0, a, c):
        """Implementation of the Gauss-Seidel relaxation"""
        c_recip = 1 / c

        for iteration in range(0, self.iter):
            # Calculates the interactions with the 4 closest neighbors
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) * c_recip

            self.set_boundaries(x)

    #Detect borders, colisions
    def set_boundaries(self, table):
        """
        Boundaries handling
        :return:
        """

        if len(table.shape) > 2:  # 3d velocity vector array
            # Simulating the bouncing effect of the velocity array
            # vertical, invert if y vector
            table[:, 0, 1] = - table[:, 0, 1]
            table[:, self.size - 1, 1] = - table[:, self.size - 1, 1]

            # horizontal, invert if x vector
            table[0, :, 0] = - table[0, :, 0]
            table[self.size - 1, :, 0] = - table[self.size - 1, :, 0]

        table[0, 0] = 0.5 * (table[1, 0] + table[0, 1])
        table[0, self.size - 1] = 0.5 * (table[1, self.size - 1] + table[0, self.size - 2])
        table[self.size - 1, 0] = 0.5 * (table[self.size - 2, 0] + table[self.size - 1, 1])
        table[self.size - 1, self.size - 1] = 0.5 * table[self.size - 2, self.size - 1] + \
                                              table[self.size - 1, self.size - 2]

        if object:
            for x in range(0, objQ):
                posTemp = objP[x].split()
                objPx = int(posTemp[0])
                objPy = int(posTemp[1])
                for i in range(objPx, objPy):
                    for j in range(objPx, objPy):
                        table[i, j]=0



    def diffuse(self, x, x0, diff):
        if diff != 0:
            a = self.dt * diff * (self.size - 2) * (self.size - 2)
            self.lin_solve(x, x0, a, 1 + 6 * a)
        else:  # equivalent to lin_solve with a = 0
            x[:, :] = x0[:, :]

    def project(self, velo_x, velo_y, p, div):
        # numpy equivalent to this in a for loop:
        # div[i, j] = -0.5 * (velo_x[i + 1, j] - velo_x[i - 1, j] + velo_y[i, j + 1] - velo_y[i, j - 1]) / self.size
        div[1:-1, 1:-1] = -0.5 * (
                velo_x[2:, 1:-1] - velo_x[:-2, 1:-1] +
                velo_y[1:-1, 2:] - velo_y[1:-1, :-2]) / self.size
        p[:, :] = 0

        self.set_boundaries(div)
        self.set_boundaries(p)
        self.lin_solve(p, div, 1, 6)

        velo_x[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.size
        velo_y[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.size

        self.set_boundaries(self.velo)

    def advect(self, d, d0, velocity):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)

        for j in range(1, self.size - 1):
            for i in range(1, self.size - 1):
                tmp1 = dtx * velocity[i, j, 0]
                tmp2 = dty * velocity[i, j, 1]
                x = i - tmp1
                y = j - tmp2

                if x < 0.5:
                    x = 0.5
                if x > (self.size - 1) - 0.5:
                    x = (self.size - 1) - 0.5
                i0 = math.floor(x)
                i1 = i0 + 1.0

                if y < 0.5:
                    y = 0.5
                if y > (self.size - 1) - 0.5:
                    y = (self.size - 1) - 0.5
                j0 = math.floor(y)
                j1 = j0 + 1.0

                s1 = x - i0
                s0 = 1.0 - s1
                t1 = y - j0
                t0 = 1.0 - t1

                i0i = int(i0)
                i1i = int(i1)
                j0i = int(j0)
                j1i = int(j1)

                try:
                    d[i, j] = s0 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                              s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
                except IndexError:
                    # tmp = str("inline: i0: %d, j0: %d, i1: %d, j1: %d" % (i0, j0, i1, j1))
                    # print("tmp: %s\ntmp1: %s" %(tmp, tmp1))
                    raise IndexError
        self.set_boundaries(d)

    def turn(self):
        self.cntx += 1
        self.cnty += 1
        if self.cntx == 3:
            self.cntx = -1
            self.rotx = 0
        elif self.cntx == 0:
            self.rotx = self.roty * -1
        if self.cnty == 3:
            self.cnty = -1
            self.roty = 0
        elif self.cnty == 0:
            self.roty = self.rotx
        return self.rotx, self.roty

def movimeinto(velX, velY, mueve, f):
    velocity = [velY,velX]
    if mueve== "circulo":
        velocity = [velY*np.sin(0.5*f), velX*np.cos(0.5*f)]
    elif mueve== "raro":
        velocity= [velY*np.sin(0.5*f)]
    else:
        print("check the name of the movement is circle or raro")
        sys.exit()
    return velocity

def ObjOB():
    objP.clear()
    print("Enter the position of the objects like an array X and Y, (12 7)")
    for x in range(0, objQ):
        pos = input()
        objP.append(pos)
    valObj()

def valObj():
    x = 0
    y = 0
    for x in range(0, objQ):
        PTemp = objP[x].split()
        x = int(PTemp[0])
        y = int(PTemp[1])

        if(x>60):
            print("enter a valid number for x position no more than 60x60")
            ObjOB()
        elif(x<0):
            print("enter a valid number for x position no more than 60x60")
            ObjOB()
        elif(y>60):
            print("enter a valid number for y position no more than 60x60")
            ObjOB()
        elif(y<0):
            print("enter a valid number for y position no more than 60x60")
            ObjOB()


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        input("press enter to start simulating!!! \(°U°)/")

        file = open("fluid_notes.txt", "r")
        ponerVal = file.readlines()

        try:
            ingreso = int(ponerVal[0])
        except:
            print("check the number of emitters")
            sys.exit()

        colores = ponerVal[len(ponerVal)-2].split()

        try:
            valorF = int(ponerVal[len(ponerVal)-1])
        except:
            print("check the number of frames")
            sys.exit()
            


        #saving the values
        for x in range(0,ingreso):
            valores.append(ponerVal[x+1])


        if len(valores)<ingreso:
            print("enter the values "+ingreso+".")
            sys.exit()
        elif len(valores)>ingreso:
            print("more values than needed,just 1"+ingreso+".")
            sys.exit()

        correctVal = False
        while correctVal==False:
            ans = input("¿add an object to the scene or not? (Y / N) ")
            if ans=="Y" or ans=="N":
                correctVal = True
                if ans=="Y":
                    object = True
                    try:
                        objQ = int(input("How many objects do you want? "))
                        if objQ>10 or objQ<0:
                            objQ = int(input("Less than 10 more than 0 "))
                        correctVal = True
                        ObjOB()
                    except ValueError:
                        objQ = int(input("Please enter a valid number"))
                        ObjOB()
                        correctVal = False
            else:
                print("enter a valid answer")
                correctVal = False

        inst = Fluid()

        def update_im(i):

            # Saving values for emitter
            # Here we can add more emitters
            for x in range(0, ingreso):
                emitter_values = valores[x].split()
                try:
                    try:
                        posX = int(emitter_values[0])
                        posY = int(emitter_values[1])
                        size = int(emitter_values[2])
                        density = int(emitter_values[3])
                        velX = int(emitter_values[4])
                        velY = int(emitter_values[5])
                    except:
                        print("check the values ")
                        sys.exit()

                    mueve = emitter_values[6]
                except IndexError:
                    print(" check the values of the emitters, 7: (positonX), (positionY), (size density), (velocityX), (velocityY), (movement)")
                    sys.exit()


                # created new density
                # it add a new density
                inst.density[posY:posY+size, posX:posX+size] += density  
                # add the velocity vector 
                inst.velo[posY, posX] = movimeinto(velX, velY, mueve, i)
                inst.step()
                im.set_array(inst.density)
                q.set_UVC(inst.velo[:, :, 1], inst.velo[:, :, 0])
                im.autoscale()

        fig = plt.figure()

        # plot density #Add color
        try:
            im = plt.imshow(inst.density, vmax=100, interpolation='bilinear', cmap=colores[0])
        except:
            print("Please check that the color name its a name from the matplotlib")
            sys.exit()

        # plot vector field
        q = plt.quiver(inst.velo[:, :, 1], inst.velo[:, :, 0], scale=10, angles='xy')
        anim = animation.FuncAnimation(fig, update_im, interval=0, frames=valorF) #How many frames will be
        #anim.save("movie.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
        anim.save("movie.mp4", fps=30)
        #plt.show()

    except ImportError:
        import imageio

        frames = 30

        flu = Fluid()

        video = np.full((frames, flu.size, flu.size), 0, dtype=float)

        for step in range(0, frames):
            flu.density[4:7, 4:7] += 100  # add density into a 3*3 square
            flu.velo[5, 5] += [1, 2]

            flu.step()
            video[step] = flu.density

        imageio.mimsave('./video.gif', video.astype('uint8'))
