"""Implementation based on the template of Matformer."""
from zeoformer.train_props import train_prop_model
props = [
    "Binding_SiO2",
    "Binding_total",
    "Templating",
    "Directivity_SiO2",
    "Competition_OSDA",
    "Competition_SiO2",
    "Competition_total",
    "Binding_OSDA",
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "spillage",
    "optb88vdw_total_energy",
    "ehull",
]
train_prop_model(learning_rate=0.001, name="zeoformer", prop=props[7], pyg_input=True, n_epochs=400, batch_size=12, output_dir="./fe_400_0.001_25_50", save_dataloader=False)