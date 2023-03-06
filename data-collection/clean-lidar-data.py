import json

jsonLidarPath = '../lidarJson.json'
with open(jsonLidarPath, 'r') as j:
    lidarData = json.loads(j.read())
    cleanAngle = [abs(a[1]-180) for a in lidarData['data']]
    cleanQuality = [q[0] for q in lidarData['data']]
    cleanDist = [d[2] for d in lidarData['data']]

    cleanData = {}
    cleanData['data'] = [[cleanQuality[i],cleanAngle[i],cleanDist[i]] for i in range(len(cleanAngle))]
    with open('../lidarJsonClean.json','w') as outfile :                    
        json.dump(cleanData, outfile, indent=2)