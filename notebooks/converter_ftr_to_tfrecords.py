import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from extract_data_from_i3files._lib.tfrecords_utils import serialize_example 
from lib.simdata_i3 import I3SimHandler

# Paths to stored files
pulse_file = "/mnt/research/IceCube/Gupta-Reco/22644/tfrecords/pulses_ds_22644_from_14000_to_15000_10_to_100TeV.ftr"
meta_file  = "/mnt/research/IceCube/Gupta-Reco/22644/tfrecords/meta_ds_22644_from_14000_to_15000_10_to_100TeV.ftr"

# Load them
df_pulses = pd.read_feather(pulse_file)
df_meta   = pd.read_feather(meta_file)

geo_file = "/mnt/home/baburish/jax/TriplePandelReco_JAX/data/icecube/detector_geometry.csv"
sim_handler = I3SimHandler(df_meta=df_meta, df_pulses=df_pulses, geo_file=geo_file)

write_path = "//mnt/research/IceCube/Gupta-Reco/22644/tfrecords/ds_22644_from_14000_to_15000_10_to_100TeV.tfrecord"
options = tf.io.TFRecordOptions(compression_type='')

with tf.io.TFRecordWriter(write_path, options) as writer:
    for i in range(len(df_meta)):
        meta, pulses = sim_handler.get_event_data(i)
        event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)

        x = event_data[['x','y','z','time','charge']].to_numpy()
        y = meta[['muon_energy_at_detector','q_tot','muon_zenith','muon_azimuth','muon_time',
                  'muon_pos_x','muon_pos_y','muon_pos_z','spline_mpe_zenith','spline_mpe_azimuth',
                  'spline_mpe_time','spline_mpe_pos_x','spline_mpe_pos_y','spline_mpe_pos_z']].to_numpy()

        writer.write(serialize_example(
            tf.constant(x, dtype=tf.float64),
            tf.constant(y, dtype=tf.float64)
        ))

