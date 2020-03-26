import bpy

walls = []
doors = []
windows = []

room_types = ['living_room', 'kitchen', 'bedroom', 'bathroom', 'closet', 
'balcony', 'corridor', 'dining_room', 'laundry_room']

wall_vertices = []
wall_faces = []
window_vertices=[]
window_faces=[]
door_vertices=[]
door_faces=[]

with open('/home/harinandan/TUM/sose2019/IDP/building-design-assistant/out/20200120_141712/out_3.txt') as vector:

    i = 0
    j = 0
    k = 0
    for line in vector:
        line = line.split(' ')
        label = line[4].strip()

        if label == 'wall':
            wall_vertices.append((int(line[0])/-10, int(line[1])/10, 0))
            wall_vertices.append((int(line[0])/-10, int(line[1])/10, 5))
            wall_vertices.append((int(line[2])/-10, int(line[3])/10, 5))
            wall_vertices.append((int(line[2])/-10, int(line[3])/10, 0))
            wall_faces.append((i, i+1, i+2, i+3))
            i += 4

        elif label == 'door':
            door_vertices.append((int(line[0])/-10-0.1, int(line[1])/10-0.1, 0))
            door_vertices.append((int(line[0])/-10-0.1, int(line[1])/10-0.1, 3))
            door_vertices.append((int(line[2])/-10-0.1, int(line[3])/10-0.1, 3))
            door_vertices.append((int(line[2])/-10-0.1, int(line[3])/10-0.1, 0))
            door_faces.append((j, j+1, j+2, j+3))
            j += 4
            door_vertices.append((int(line[0])/-10+0.1, int(line[1])/10+0.1, 0))
            door_vertices.append((int(line[0])/-10+0.1, int(line[1])/10+0.1, 3))
            door_vertices.append((int(line[2])/-10+0.1, int(line[3])/10+0.1, 3))
            door_vertices.append((int(line[2])/-10+0.1, int(line[3])/10+0.1, 0))
            door_faces.append((j, j+1, j+2, j+3))
            j += 4
        elif label == 'window':
            window_vertices.append((int(line[0])/-10-0.1, int(line[1])/10-0.1, 2))
            window_vertices.append((int(line[0])/-10-0.1, int(line[1])/10-0.1, 4))
            window_vertices.append((int(line[2])/-10-0.1, int(line[3])/10-0.1, 4))
            window_vertices.append((int(line[2])/-10-0.1, int(line[3])/10-0.1, 2))
            window_faces.append((k, k+1, k+2, k+3))
            k += 4
            window_vertices.append((int(line[0])/-10+0.1, int(line[1])/10+0.1, 2))
            window_vertices.append((int(line[0])/-10+0.1, int(line[1])/10+0.1, 4))
            window_vertices.append((int(line[2])/-10+0.1, int(line[3])/10+0.1, 4))
            window_vertices.append((int(line[2])/-10+0.1, int(line[3])/10+0.1, 2))
            window_faces.append((k, k+1, k+2, k+3))
            k += 4


mymesh = bpy.data.meshes.new("Walls")
myobject = bpy.data.objects.new("Walls", mymesh)

myobject.location = bpy.context.scene.cursor.location
bpy.context.scene.collection.objects.link(myobject)

mymesh.from_pydata(wall_vertices, [], wall_faces)
mymesh.update(calc_edges=True)


mymesh = bpy.data.meshes.new("Doors")
myobject = bpy.data.objects.new("Doors", mymesh)

myobject.location = bpy.context.scene.cursor.location
bpy.context.scene.collection.objects.link(myobject)

mymesh.from_pydata(door_vertices, [], door_faces)
mymesh.update(calc_edges=True)


mymesh = bpy.data.meshes.new("Windows")
myobject = bpy.data.objects.new("Windows", mymesh)

myobject.location = bpy.context.scene.cursor.location
bpy.context.scene.collection.objects.link(myobject)

mymesh.from_pydata(window_vertices, [], window_faces)
mymesh.update(calc_edges=True)