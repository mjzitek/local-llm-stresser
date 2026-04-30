import json, os, sys

def process_data(d):
    r = []
    for x in d:
        if x.get('active') == True:
            if x.get('score') != None:
                if x['score'] > 50:
                    nm = x.get('name', 'unknown')
                    sc = x['score']
                    cat = ''
                    if sc > 90: cat = 'A'
                    elif sc > 75: cat = 'B'
                    elif sc > 60: cat = 'C'
                    else: cat = 'D'
                    r.append({'name': nm, 'score': sc, 'category': cat})
    r2 = []
    seen = []
    for it in r:
        if it['name'] not in seen:
            seen.append(it['name'])
            r2.append(it)
    r2.sort(key=lambda z: z['score'], reverse=True)
    return r2

def main():
    if len(sys.argv) < 2:
        print('usage: ...'); sys.exit(1)
    f = sys.argv[1]
    if os.path.exists(f) == False:
        print('no file'); sys.exit(2)
    fp = open(f); data = json.load(fp); fp.close()
    out = process_data(data)
    for o in out:
        print(o['name'], o['score'], o['category'])

if __name__ == '__main__':
    main()
