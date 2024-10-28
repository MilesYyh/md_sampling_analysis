#######################################################################################
"""
@Miles
@Beijing 26/10/2024 
@todo: Add function to select the scpecific pose that is the most populated. Add the color design.
"""
#######################################################################################

import os
import sys
import numpy as np
import pandas as pd
# import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns 
# from scipy.stats import gaussian_kde
# from joblib import Parallel, delayed
import pytraj as pt


class md_density():

    def __init__(self,
                    traj,top,time,
                    calc_obj,mask1,mask2,cv_names,
                    kde,bins,figure_size):  

        self.traj = traj
        self.top = top
        self.time = time 
        self.calc_obj = calc_obj
        self.mask1 = mask1
        self.mask2 = mask2
        self.cv_names = cv_names
        self.kde = kde
        self.bins = bins
        self.figure_size = figure_size
        
    #load traj and top into the "u" object 
    def load_data(self):
            self.u = pt.iterload(self.traj,self.top)

    def time_scale_generate(self):
        if self.time is not None:
            result = np.arange(self.time,self.u.n_frames*self.time+self.time,self.time)
        else:
            result = np.arange(1,self.u.n_frames+1,1)
        return result



    #calculate specific data of two masks (mask1,mask2) from the trajectory 
    def md_density_masks(self):
        # data calculate within pytraj
        if self.calc_obj == "distance":
            data_mask1 = pt.distance(self.u,mask=self.mask1);data_mask2 = pt.distance(self.u,mask=self.mask2)    
        elif self.calc_obj == "angle":
            data_mask1 = pt.angle(self.u,mask=self.mask1);data_mask2 = pt.angle(self.u,mask=self.mask2)    
        elif self.calc_obj == "dihedral":
            data_mask1 = pt.dihedral(self.u,mask=self.mask1);data_mask2 = pt.dihedral(self.u,mask=self.mask2)    
       
        time_x = self.time_scale_generate()
        data = np.vstack((time_x,data_mask1,data_mask2)).T

        #save data into files     
        filename = str(self.calc_obj)+'_data.tsv'
        np.savetxt(filename, 
                   data, 
                   delimiter='\t', 
                   fmt=['%.4f', '%.6f', '%.6f'], 
                   header='time\t'+self.cv_names[0]+'\t'+self.cv_names[1],
                   comments=''
                   )
        print(f"{self.calc_obj} data saved in '{filename}'")

        
        fig_combined, axs_combined = plt.subplots(1,1, figsize=(6, 6), sharey=True)  
        color= ["black","grey"] 
        if self.kde is not None:
            #density generate by kde
            sns.kdeplot(data=data[:,1],label=self.cv_names[0],color=color[0])
            sns.kdeplot(data=data[:,2],label=self.cv_names[1],color=color[1])
        else:
            #density generate by pdf
            for data_column in range(1,3):
                #probability distrbution function
                hist, bin_edges = np.histogram(data[:,data_column], bins=self.bins, density=True)
                # hist = np.clip(hist, a_min=1e-10, a_max=None)
                # G = -self.boltzmann_constant * self.temperature * np.log(hist)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                # G_min_normalized = G - np.min(G)
                hist = hist / hist.sum()
                axs_combined.plot(bin_centers,hist, 
                        color=color[data_column-1], label=self.cv_names[data_column-1])
                
                # data_to_save = np.vstack([bin_edges[:100],hist]).T
                # filename = str(self.calc_obj)+str(self.cv_names[data_column-1])+'_dist.tsv'
                # np.savetxt(filename, 
                #         data_to_save, 
                #         delimiter='\t', 
                #         fmt=['%.4f', '%.6f'], 
                #         header=str(self.calc_obj)+'\t'+str(self.cv_names[data_column-1]),
                #         comments=''
                #        )
        axs_combined.legend(loc='upper right')
        axs_combined.set_ylabel('Normalised Population')
        axs_combined.set_xlabel(self.calc_obj)
        # plt.savefig(str(self.calc_obj)+'_density.png')
        plt.savefig(str(self.calc_obj)+'_density.tiff')
        plt.show()
    

    #calculate specific data of one mask (mask1) from the trajectory 
    def md_density_mask(self):
        #data calculate within pytraj
        if self.calc_obj == "distance" and self.u is not None:
            data_mask1 = pt.distance(self.u,mask=self.mask1);    
        elif self.calc_obj == "angle" and self.u is not None:
            data_mask1 = pt.angle(self.u,mask=self.mask1);    
        elif self.calc_obj == "dihedral" and self.u is not None:
            data_mask1 = pt.dihedral(self.u,mask=self.mask1);    
        time_x = self.time_scale_generate()
        data = np.vstack((time_x,data_mask1)).T
        # print(data.shape)

        #save data into files     
        filename = str(self.calc_obj)+'_data.tsv'
        np.savetxt(filename, 
                   data, 
                   delimiter='\t', 
                   fmt=['%.4f', '%.6f'], 
                   header='time\t'+str(self.calc_obj),
                   comments=''
                   )
        print(f"{self.calc_obj} data saved in '{filename}'")

        fig_combined, axs_combined = plt.subplots(1,1, figsize=(6, 6), sharey=True)  
        color= ["black","grey"] 
        if self.kde is not None:
            #density generate by kde
            sns.kdeplot(data=data[:,1],label=self.cv_names[0],color=color[0])
        else:
            #density generate by pdf
            hist, bin_edges = np.histogram(data[:,1], bins=self.bins, density=True)
            # hist = np.clip(hist, a_min=1e-10, a_max=None)
            # G = -self.boltzmann_constant * self.temperature * np.log(hist)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # G_min_normalized = G - np.min(G)
            hist = hist / hist.sum()
            axs_combined.plot(bin_centers,hist, 
                    color=color[0], label=self.cv_names[0])
        axs_combined.legend(loc='upper right')
        axs_combined.set_ylabel('Normalised Population')
        axs_combined.set_xlabel(self.calc_obj)
        # plt.savefig(str(self.calc_obj)+'_density.png')
        plt.savefig(str(self.calc_obj)+'_density.tiff')
        plt.show()

    def figure_style_init(self):
        plt.rcParams["figure.dpi"] = 600               
        plt.rcParams["font.family"]="Times New Roman"  
        plt.rcParams["font.style"]="normal"            
        plt.rcParams["font.weight"]=400                
        plt.rcParams["font.size"]=10                   
        plt.rcParams["lines.linewidth"]=2              
        plt.rcParams["axes.linewidth"]=2               
        plt.rcParams["axes.labelsize"]=12              
        plt.rcParams["axes.labelpad"]=1                
        plt.rcParams["axes.labelweight"]=600           
        plt.rcParams["axes.labelcolor"]="k"            
        plt.rcParams["axes.spines.left"]="True"        
        plt.rcParams["axes.spines.bottom"]="True"      
        plt.rcParams["axes.spines.top"]="True"         
        plt.rcParams["axes.spines.right"]="True"       
        plt.rcParams["xtick.major.width"]=2            
        plt.rcParams["xtick.minor.width"]=1            
        plt.rcParams["xtick.labelsize"]=12             
        plt.rcParams["xtick.minor.visible"]="True"     
        plt.rcParams["ytick.major.width"]=2            
        plt.rcParams["ytick.minor.width"]=1            
        plt.rcParams["ytick.labelsize"]=12             
        plt.rcParams["ytick.minor.visible"]="True"     
        plt.rcParams["legend.frameon"]="False"         
        plt.rcParams["legend.framealpha"]=0.8          
        plt.rcParams["legend.fontsize"]=12             
        plt.rcParams["legend.columnspacing"]=1.0       

    def main(self):
        print("Loading data...")
        self.load_data()
        print("Data loaded successfully!")
        print(f"traj: {self.traj}\ntop: {self.top}\n")

        self.figure_style_init()
        print(f"Calculate the {self.calc_obj} data...\n")
        if self.mask1 is not None and self.mask2 is not None:
            self.md_density_masks(
                )
        elif self.mask1 is not None and self.mask2 is None:
            self.md_density_mask(
                #test
                )
        print("Plot successfully generated.\n")


    @staticmethod
    def help():
        help_text = """
        Optional arguments:
            --time                  [int]       the every step (unit ps) of your trajectory (default None)
            --calc_obj              [str]       the data u want to calculate <ditance|angle|diherdral> (default distance)
            --mask1                 [str]       the group u want to calculate by the calc_obj  (default None)
            --mask2                 [str]       the group u want to calculate by the calc_obj  (default None)
            --cv_names              [str] [str] Names for the collective variables (default: CV1, CV2)
            --kde                   [str]       use the gaussian_kde (default True)
            --bins                  [int]       Bins for histogram (default: 100)
            --figure_size           [int] [int] (default: 6 6)
            --color                 [str]       the line color 
            
        Example:
            python density.py test.nc test.prmtop --time 0.005 --mask1 "@8049 @8028" --mask2 "@8050 @8028"  
        """
        print(help_text)

def main():
    #define the paras 
    time     = None             # --time    
    calc_obj = "distance"       # --calc_obj      
    mask1 = None                # --mask1
    mask2 = None                # --mask2
    cv_names = ['CV1', 'CV2']   # --cv_names     
    kde = True                  # --kde
    bins = 100                  # --bins_histogram 
    figure_size = (6,6)         # --figure_size
 

    if len(sys.argv) >= 3:
        traj, top = sys.argv[1], sys.argv[2]
        i = 3   
        while i < len(sys.argv):
            key = sys.argv[i]
            if key == "--time":
                time = float(sys.argv[i + 1])
                i += 2
            elif key == "--calc_obj":
                calc_obj = sys.argv[i + 1]
                i += 2
            elif key == "--mask1":
                mask1 = sys.argv[i + 1]
                i += 2
            elif key == "--mask2":
                mask2 = sys.argv[i + 1]
                i += 2
            elif key == "--cv_names":
                cv_names = [sys.argv[i + 1], sys.argv[i + 2]]
                i += 3
            elif key == "--kde":
                kde = sys.argv[i + 1]
                i += 2
            elif key == "--bins":
                bins_histogram = int(sys.argv[i + 1])
                i += 2
            elif key == "--figure_size":
                figure_size = (int(sys.argv[i+1]),int(sys.argv[i+1]))
                i += 3
            else:
                print(f"Unrecognized option: {key}")
                sys.exit(1)
    else:
        md_density.help()
        sys.exit(1)

    try:
        fel = md_density(traj,top,time=time,
                         calc_obj=calc_obj,mask1=mask1,mask2=mask2,cv_names=cv_names, 
                         kde=kde,bins=bins,figure_size=figure_size
                        )
        fel.main()
        


    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

        
        