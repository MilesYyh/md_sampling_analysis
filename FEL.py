#######################################################################################
"""
@Miles
@Beijing 20/10/2024 
@This scripts I modified from https://github.com/sulfierry/free_energy_landscape/
@todo:The figure style need to more detail.
"""
#######################################################################################

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
# import time

class FreeEnergyLandscape:

    def __init__(self, cv1_path, cv2_path, 
                 temperature, boltzmann_constant, 
                 bins=100, kde_bandwidth=None, 
                 cv_names=['CV1', 'CV2'], discrete=None,
                 xlim_inf=None, xlim_sup=None, ylim_inf=None, ylim_sup=None,
                 num_cpus=16):  

        
        self.cv1_path = cv1_path
        self.cv2_path = cv2_path
        self.temperature = temperature
        self.kB = boltzmann_constant
        self.cv_names = cv_names
        self.num_cpus = num_cpus
        self.colors = [
            (0.0, "darkblue"),
            (0.1, "blue"),
            (0.2, "dodgerblue"),
            (0.3, "deepskyblue"),
            (0.4, "lightblue"),
            (0.5, "azure"),
            (0.6, "oldlace"),
            (0.7, "wheat"),
            (0.8, "lightcoral"),
            (0.9, "indianred"),
            (1.0, "darkred")
        ]
        self.custom_cmap = LinearSegmentedColormap.from_list(
            "custom_energy", 
            self.colors
            )
        
        self.proj1_data_original = None
        self.proj2_data_original = None
        self.bins = bins
        self.kde_bandwidth = kde_bandwidth
        self.discrete = discrete
        self.discreet_colors = [
            'purple', 
            'darkorange', 
            'green', 
            'lightgrey', 
            'red',
            'magenta',
            'mediumorchid',
            'deeppink',
            'peru',
            'indianred'
            ]
    
        self.discreet_markers = [
            '*', 
            's', 
            '^', 
            'D', 
            'o',
            'p',
            'h',
            'v',
            'X',
            'd'
            ]

        self.xlim_inf = xlim_inf
        self.xlim_sup = xlim_sup
        self.ylim_inf = ylim_inf
        self.ylim_sup = ylim_sup


    #load the reaction coordiantes data.
    def load_data(self):
        self.proj1_data_original = np.loadtxt(self.cv1_path, usecols=[1])
        self.proj2_data_original = np.loadtxt(self.cv2_path, usecols=[1])
        tmp = np.vstack([self.proj1_data_original,self.proj2_data_original]).T
        self.pd_data = pd.DataFrame(tmp,columns=[self.cv_names[0],self.cv_names[1]])


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

    def plot_jointplot(self):
        plt.figure(figsize=(6, 6))
        sns.jointplot(
            data=self.pd_data,
            x=self.cv_names[0],
            y=self.cv_names[1],
            kind="kde",
            height=5,
            ratio=10,
        )

        plt.xlabel(self.cv_names[0])
        plt.ylabel(self.cv_names[1])
        plt.savefig("jointpot.png", dpi=600, bbox_inches="tight")
        plt.close()


    #calculate the Gibss Energy with G(x) formula within the kde probability
    def calculate_free_energy(self, data):
        if hasattr(self, 'cached_results'):
            return self.cached_results

        values_original = np.vstack([data[:, 0], data[:, 1]]).T
        if self.kde_bandwidth:
            kernel_original = gaussian_kde(values_original.T, bw_method=self.kde_bandwidth)
        else:
            kernel_original = gaussian_kde(values_original.T)

        # Ajusta a geração da grade para respeitar os limites especificados
        x_min = self.xlim_inf if self.xlim_inf is not None else data[:, 0].min()
        x_max = self.xlim_sup if self.xlim_sup is not None else data[:, 0].max()
        y_min = self.ylim_inf if self.ylim_inf is not None else data[:, 1].min()
        y_max = self.ylim_sup if self.ylim_sup is not None else data[:, 1].max()
        
        X_original, Y_original = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions_original = np.vstack([X_original.ravel(), Y_original.ravel()])
        Z_original = np.reshape(kernel_original(positions_original).T, X_original.shape)
        G_original = -self.kB * self.temperature * np.log(Z_original)
        G_original = np.clip(G_original - np.min(G_original), 0, 25)

        self.cached_results = {'X_original': X_original, 
                            'Y_original': Y_original, 
                            'G_original': G_original
                            }
                            
        #to save the data that use to make FEL.png -> .dat&.npy 
        np.save("FEL_plot_data.npy",self.cached_results)
        csv_X_original = self.cached_results["X_original"].flatten()
        csv_Y_original = self.cached_results["Y_original"].flatten()
        csv_G_original = self.cached_results["G_original"].flatten()
        csv_data = np.vstack([csv_X_original,csv_Y_original,csv_G_original]).T
        # np.savetxt("FEL_plot_data.dat", 
        #     csv_data, 
        #     delimiter=' ', 
        #     fmt=['%6.6f', '%6.6f', '%6.6f'], 
        #     header= '%13s%16s%16s' % ('X_original','Y_original','G_original'), 
        #     comments=''
        #     )
        # np.savetxt("FEL_plot_data.tsv", 
        np.savetxt("FEL_plot_data.tsv", 
                   csv_data, 
                   delimiter='\t', 
                   fmt=['%6.6f', '%6.6f', '%6.6f'],
                   header= 'X_original\tY_original\tG_original', 
                   comments=''
                   )
        return self.cached_results

    def plot_energy_landscape(self, threshold, titles=['CV1', 'CV2'], xlim_inf=None, xlim_sup=None, ylim_inf=None, ylim_sup=None):

        if xlim_inf is not None and xlim_sup is not None:
            plt.xlim(xlim_inf, xlim_sup)
        if ylim_inf is not None and ylim_sup is not None:
            plt.ylim(ylim_inf, ylim_sup)

        data = np.hstack((self.proj1_data_original[:, None], self.proj2_data_original[:, None]))
        result = self.calculate_free_energy(data)
        plt.figure(figsize=(8, 6))

        custom_cmap = LinearSegmentedColormap.from_list("custom_energy", self.colors)

        G_min, G_max = np.min(result['G_original']), np.max(result['G_original'])
        levels = np.arange(G_min, G_max, 2)
        
        cont = plt.contourf(result['X_original'], result['Y_original'], result['G_original'],
                            levels=levels, cmap=custom_cmap, extend='both')
        plt.contour(result['X_original'], result['Y_original'], result['G_original'],
                    levels=levels, colors='k', linewidths=0.5)
        if self.discrete is not None and threshold is not None:
            discrete_intervals = np.arange(0, threshold, self.discrete)

            for i, interval in enumerate(discrete_intervals):
                end = min(interval + self.discrete, threshold)
                mask = (result['G_original'].flatten() <= end) & (result['G_original'].flatten() > interval)
                X_flat, Y_flat = result['X_original'].flatten(), result['Y_original'].flatten()

                if np.any(mask):
                    plt.scatter(X_flat[mask], 
                                Y_flat[mask], 
                                color=self.discreet_colors[i % len(self.discreet_colors)],
                                marker=self.discreet_markers[i % len(self.discreet_markers)], 
                                label=f'{interval:.1f}-{end:.1f} KJ/mol'
                                )

        if threshold is not None:
            plt.legend(loc='lower left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        cbar = plt.colorbar(cont)
        cbar.set_label('Free energy (kJ/mol)')
        plt.xlim(self.xlim_inf, self.xlim_sup)
        plt.ylim(self.ylim_inf, self.ylim_sup)
        plt.xlabel(titles[0])
        plt.ylabel(titles[1])
        # plt.title('Free Energy Landscape')
        # plt.title('Gbbis Energy Landscape')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        plt.savefig("FEP.png")
        plt.show()


    def plot_threshold_points(self, ax, result, lower_bound, upper_bound, color, label):
        G_flat = result['G_original'].flatten()
        energy_mask = (G_flat >= lower_bound) & (G_flat < upper_bound)

        if any(energy_mask):
            X_flat, Y_flat = result['X_original'].flatten(), result['Y_original'].flatten()
            ax.scatter(X_flat[energy_mask], Y_flat[energy_mask], G_flat[energy_mask], color=color, s=20, label=label)


    def calculate_density_for_chunk(self, combined_data_chunk, bw_method):
        # Esta função é uma versão simplificada que recalcula o kernel para cada chunk
        kernel = gaussian_kde(combined_data_chunk.T, bw_method=bw_method)
        density = np.exp(kernel.logpdf(combined_data_chunk.T))
        return density

    def calculate_and_save_free_energy(self, threshold=None):

        # Verifica se os dados foram carregados
        if self.proj1_data_original is None or self.proj2_data_original is None:
            raise ValueError("Data not loaded. Run load_data first.")

        # Carrega os índices dos frames
        frames = np.loadtxt(self.cv1_path, usecols=[0], dtype=np.float64).astype(np.int64)

        # Prepara os dados combinados
        combined_data = np.vstack((self.proj1_data_original, self.proj2_data_original)).T

        # num_cpus = multiprocessing.cpu_count()
        # num_cpus = self.num_cpus
        print(f"The num of cpu cores:{self.num_cpus}")
        data_chunks = np.array_split(combined_data, self.num_cpus, axis=0)

        # Recalcula a densidade de probabilidade para cada chunk de dados em paralelo
        results = Parallel(n_jobs=self.num_cpus)(delayed(self.calculate_density_for_chunk)(chunk, self.kde_bandwidth) for chunk in data_chunks)
        density = np.concatenate(results)
        
        # Calcula a energia livre
        G = -self.kB * self.temperature * np.log(density)
        G_min = np.min(G)
        G_normalized = G - G_min

        # Aplica o threshold, se especificado
        if threshold is not None:
            indices_below_threshold = G_normalized <= threshold
            filtered_frames = frames[indices_below_threshold]
            filtered_cv1 = self.proj1_data_original[indices_below_threshold]
            filtered_cv2 = self.proj2_data_original[indices_below_threshold]
            filtered_energy = G_normalized[indices_below_threshold]
        else:
            filtered_frames = frames
            filtered_cv1 = self.proj1_data_original
            filtered_cv2 = self.proj2_data_original
            filtered_energy = G_normalized

        # Prepara os dados para salvamento
        data_to_save = np.column_stack((filtered_frames, filtered_cv1, filtered_cv2, filtered_energy))

        # Ordena os dados pela energia
        data_to_save = data_to_save[data_to_save[:, 3].argsort()]

        # filename = 'discrete_values_energy_frames.dat'
        # np.savetxt(filename, 
        #            data_to_save, 
        #            delimiter=' ', 
        #            fmt=['%11d', '%6.6f', '%6.6f', '%6.6f'], 
        #            header= '%s%9s%9s%12s' % ('#Frame','cv1','cv2','energy'), 
        #            comments=''
        #            )        

        filename = 'FEL_energy_discretize_frames.tsv'
        np.savetxt(filename, 
                   data_to_save, 
                   delimiter='\t', 
                   fmt=['%d', '%.6f', '%.6f', '%.6f'], 
                   header='frame\tcv1\tcv2\tenergy', 
                   comments=''
                   )
        print(f"Energy data saved in '{filename}'")


    def main(self, energy_threshold, cv_names):
        print("Loading data...")
        self.load_data()
        print("Data loaded successfully!")
        print(f"CV1: {self.cv1_path}\nCV2: {self.cv2_path}\n")
        # with open(log_file_name,"a+") as fp:
        #     fp.write("%s\n%s" % (self.cv1_path,self.cv2_path));fp.close()

        self.figure_style_init()
        print("Plotting the CVs density...\n")
        self.plot_jointplot()
        print("Plotting the free energy landscape...\n")
        self.plot_energy_landscape(
            threshold=energy_threshold, titles=cv_names
            )
        print("Plot successfully generated.\n")

        # Após o uso final dos dados, limpe-os para liberar memória
        if hasattr(self, 'cached_results'):
            del self.cached_results

    @staticmethod
    def help():
        help_text = """
        Optional arguments:
            --temperature           [int]       Simulation temperature in Kelvin (default: 300K)
            --kb                    [float]     Boltzmann constant in kJ/(mol·K) (default: 8.314e-3)
            --energy                [float]     Energy (KJ/mol), single value (default: None)
            --discretize            [float]     Discrete value (KJ/mol) for energy (default: None)
            --bins_energy_histogram [int]       Bins for energy histogram (default: 100)
            --kde_bandwidth         [float]     Bandwidth for kernel density estimation (default: None)
            --names                 [str] [str] Names for the collective variables (default: CV1, CV2)
            --xlim_inf              [float]     Lower limit for the x-axis (default: None)
            --xlim_sup              [float]     Upper limit for the x-axis (default: None)
            --ylim_inf              [float]     Lower limit for the y-axis (default: None)
            --ylim_sup              [float]     Upper limit for the y-axis (default: None)
            --num_cpus              [int]       The cpu core num to use (default 16)

        Example:
            python FEL.py test1.txt test2.txt --energy 3.0 --discretize 1.0  --names Angle Distance
            python FEL.py test1.txt test2.txt --energy 3.0 --discretize 1.0 --xlim_inf 27 --xlim_sup 130 --ylim_inf 3 --ylim_sup 37 
        

        """
        print(help_text)

def main():
    # Definindo valores padrão
    t = 300                     # --temperature           [int] [Kelvin]
    kB = 8.314e-3               # --kb                    [float] [kJ/(mol·K)]
    energy_threshold = None     # --energy                [float] [kJ/mol]
    bins_energy_histogram = 100 # --bins_energy_histogram [int]
    kde_bandwidth_cv = None     # --kde_bandwidth         [float]
    cv_names = ['CV1', 'CV2']   # --name                  [str] [str]
    discrete_val = None         # --discrete              [float]
    xlim_inf = xlim_sup = ylim_inf = ylim_sup = None  # Inicialização padrão
    num_cpus = 16


    if len(sys.argv) >= 3:
        cv1_path, cv2_path = sys.argv[1], sys.argv[2]
        # Processar argumentos adicionais como pares chave-valor
        i = 3   
        while i < len(sys.argv):
            key = sys.argv[i]
            if key == "--temperature":
                t = float(sys.argv[i + 1])
                i += 2
            elif key == "--kb":
                kB = float(sys.argv[i + 1])
                i += 2
            elif key == "--energy":
                energy_threshold = float(sys.argv[i + 1])
                i += 2
            elif key == "--discretize":
                discrete_val = float(sys.argv[i + 1])  
                i += 2
            elif key == "--bins_energy_histogram":
                bins_energy_histogram = int(sys.argv[i + 1])
                i += 2
            elif key == "--kde_bandwidth":
                kde_bandwidth_cv = float(sys.argv[i + 1]) if sys.argv[i + 1].lower() != "none" else None
                i += 2
            elif key == "--names":
                cv_names = [sys.argv[i + 1], sys.argv[i + 2]]
                i += 3
            elif key == "--xlim_inf":
                xlim_inf = float(sys.argv[i + 1])
                i += 2
            elif key == "--xlim_sup":
                xlim_sup = float(sys.argv[i + 1])
                i += 2
            elif key == "--ylim_inf":
                ylim_inf = float(sys.argv[i + 1])
                i += 2
            elif key == "--ylim_sup":
                ylim_sup = float(sys.argv[i + 1])
                i += 2
            elif key == "--num_cpus":
                num_cpus = int(sys.argv[i + 1])
                i += 2

            else:
                print(f"Unrecognized option: {key}")
                sys.exit(1)
    else:
        FreeEnergyLandscape.help()
        sys.exit(1)

    try:
        fel = FreeEnergyLandscape(cv1_path, cv2_path, t, kB, 
                                bins=bins_energy_histogram, 
                                kde_bandwidth=kde_bandwidth_cv, 
                                cv_names=cv_names, 
                                discrete=discrete_val,
                                xlim_inf=xlim_inf, xlim_sup=xlim_sup, 
                                ylim_inf=ylim_inf, ylim_sup=ylim_sup,
                                num_cpus=num_cpus)

        fel.main(energy_threshold, cv_names=cv_names)
        
        if energy_threshold is not None:
            print("Calculating and saving energy for each frame...")
            fel.calculate_and_save_free_energy(threshold=energy_threshold)
            print("Energy saved successfully!\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
