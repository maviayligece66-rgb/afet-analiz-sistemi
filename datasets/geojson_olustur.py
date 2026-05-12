import json
import os

iller = {
    "adana": [35.3213, 37.0000],
    "adiyaman": [38.2763, 37.7648],
    "afyonkarahisar": [30.5387, 38.7569],
    "agri": [43.0503, 39.7191],
    "amasya": [35.8331, 40.6499],
    "ankara": [32.8597, 39.9334],
    "antalya": [30.7133, 36.8969],
    "artvin": [41.8194, 41.1828],
    "aydin": [27.8456, 37.8560],
    "balikesir": [27.8826, 39.6484],
    "bilecik": [29.9792, 40.1501],
    "bingol": [40.4983, 38.8853],
    "bitlis": [42.1095, 38.4006],
    "bolu": [31.6061, 40.7395],
    "burdur": [30.2908, 37.7203],
    "bursa": [29.0601, 40.1828],
    "canakkale": [26.4086, 40.1553],
    "cankiri": [33.6134, 40.6013],
    "corum": [34.9537, 40.5506],
    "denizli": [29.0963, 37.7765],
    "diyarbakir": [40.2306, 37.9144],
    "edirne": [26.5557, 41.6771],
    "elazig": [39.2264, 38.6748],
    "erzincan": [39.4902, 39.7500],
    "erzurum": [41.2769, 39.9000],
    "eskisehir": [30.5256, 39.7767],
    "gaziantep": [37.3833, 37.0662],
    "giresun": [38.3874, 40.9128],
    "gumushane": [39.4814, 40.4603],
    "hakkari": [43.7408, 37.5744],
    "hatay": [36.2023, 36.4018],
    "isparta": [30.5566, 37.7648],
    "mersin": [34.6415, 36.8121],
    "istanbul": [28.9784, 41.0082],
    "izmir": [27.1428, 38.4237],
    "kars": [43.0975, 40.6013],
    "kastamonu": [33.7753, 41.3887],
    "kayseri": [35.4826, 38.7205],
    "kirklareli": [27.2252, 41.7351],
    "kirsehir": [34.1600, 39.1425],
    "kocaeli": [29.9195, 40.8533],
    "konya": [32.4846, 37.8746],
    "kutahya": [29.9833, 39.4167],
    "malatya": [38.3552, 38.3554],
    "manisa": [27.4265, 38.6191],
    "kahramanmaras": [36.9371, 37.5753],
    "mardin": [40.7351, 37.3212],
    "mugla": [28.3665, 37.2153],
    "mus": [41.4910, 38.9462],
    "nevsehir": [34.6857, 38.6244],
    "nigde": [34.6793, 37.9667],
    "ordu": [37.8797, 40.9862],
    "rize": [40.5177, 41.0201],
    "sakarya": [30.4033, 40.7569],
    "samsun": [36.3361, 41.2867],
    "siirt": [41.9458, 37.9333],
    "sinop": [35.1531, 42.0264],
    "sivas": [37.0150, 39.7477],
    "tekirdag": [27.5117, 40.9780],
    "tokat": [36.5544, 40.3167],
    "trabzon": [39.7168, 41.0015],
    "tunceli": [39.5483, 39.1062],
    "sanliurfa": [38.7955, 37.1674],
    "usak": [29.4058, 38.6823],
    "van": [43.3800, 38.5012],
    "yozgat": [34.8147, 39.8181],
    "zonguldak": [31.7987, 41.4564],
    "aksaray": [34.0254, 38.3687],
    "bayburt": [40.2249, 40.2552],
    "karaman": [33.2150, 37.1759],
    "kirikkale": [33.5153, 39.8468],
    "batman": [41.1293, 37.8812],
    "sirnak": [42.4594, 37.4187],
    "bartin": [32.3375, 41.5811],
    "ardahan": [42.7022, 41.1105],
    "igdir": [44.0450, 39.9237],
    "yalova": [29.2769, 40.6549],
    "karabuk": [32.6277, 41.2061],
    "kilis": [37.1147, 36.7184],
    "osmaniye": [36.2478, 37.0742],
    "duzce": [31.1626, 40.8438]
}

features = []

for il, koordinat in iller.items():
    lon, lat = koordinat
    fark = 0.25

    feature = {
        "type": "Feature",
        "properties": {
            "name": il
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon - fark, lat + fark],
                [lon + fark, lat + fark],
                [lon + fark, lat - fark],
                [lon - fark, lat - fark],
                [lon - fark, lat + fark]
            ]]
        }
    }

    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

os.makedirs("datasets", exist_ok=True)

with open("datasets/turkey_provinces.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=4)

print("turkey_provinces.geojson oluşturuldu.")
