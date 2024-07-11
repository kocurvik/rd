import seaborn as sns

experiments = ['Feq_7pt', 'Feq_7pt_s3', 'kFk_8pt', 'kFk_9pt', 'k2k1_9pt', 'k2Fk1_10pt', 'F_7pt', 'F_7pt_s3', 'Fns_7pt']
iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
colors = {exp: sns.color_palette("tab10")[i] for i, exp in enumerate(experiments)}
colors[experiments[7]] = sns.color_palette("tab10")[9]
basenames_pt = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
                'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', 'reichstag',
                'sacre_coeur', 'st_peters_square', 'taj_mahal', 'temple_nara_japan', 'trevi_fountain']
# basenames_eth = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'list.py', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
basenames_eth = ['courtyard', 'delivery_area', 'electro', 'kicker', 'meadow', 'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
