import requests

data = {
        'shipping': 0,
        'item_condition': 3,
        'brand_name': 'razer',
        'gen_cat': 'Electronic',
        'sub1_cat': 'Computers & Tablet'
        }

res = requests.post('https://api-ywnvzro66q-ew.a.run.app/price', json=data)

print(res.text)