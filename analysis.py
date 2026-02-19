import numpy as np
import pickle

file_name = 'logs/ode_noisy_acc/TEnd1p3_1p7_dopri5_Noise_Inject_KD_CRD_PCN_5NoiseLevel.pkl'

with open(file_name, 'rb') as f:
    acc_dict = pickle.load(f)

# Extract results organized by t_end and noise level
t_end_list = ['1.3', '1.5', '1.7']
noise_levels = [0, 0.1, 0.2, 0.3, 0.4]

print("Results organized by t_end and noise level:")
print("=" * 80)

for t_end in t_end_list:
    if t_end not in acc_dict:
        continue
    
    print(f"\nt_end = {t_end}:")
    print("-" * 80)
    
    data = acc_dict[t_end]
    if 'noise_acc_spec' not in data:
        continue
    
    johnson_data = data['noise_acc_spec'].get('Johnson', {})
    
    for noise_level in noise_levels:
        if noise_level not in johnson_data:
            continue
        
        acc_list = np.array(johnson_data[noise_level])
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list, ddof=0)  # population std
        
        print(f"  Noise {noise_level:4.1f}: {mean_acc:6.2f}±{std_acc:5.2f}")

print("\n" + "=" * 80)