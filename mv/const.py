from collections import OrderedDict

objects = OrderedDict([
    # materials
    ("none-material",    { "index": 0,  "weight": 1 }),
    ("water",            { "index": 1,  "weight": 1 }),
    ("grass",            { "index": 2,  "weight": 1 }),
    ("stone",            { "index": 3,  "weight": 1 }),
    ("path",             { "index": 4,  "weight": 1 }),
    ("sand",             { "index": 5,  "weight": 1 }),
    ("tree",             { "index": 6,  "weight": 1 }),
    ("lava",             { "index": 7,  "weight": 1 }),
    ("coal",             { "index": 8,  "weight": 1 }),
    ("iron",             { "index": 9,  "weight": 1 }),
    ("diamond",          { "index": 10, "weight": 1 }),
    ("table",            { "index": 11, "weight": 1 }),
    ("furnace",          { "index": 12, "weight": 1 }),

    # objects
    ("none-object",      { "index": 13, "weight": 1 }),
    ('player-sleep',     { "index": 14, "weight": 1 }),
    ('player-left',      { "index": 15, "weight": 1 }),
    ('player-right',     { "index": 16, "weight": 1 }),
    ('player-up',        { "index": 17, "weight": 1 }),
    ('player-down',      { "index": 18, "weight": 1 }),
    ("cow",              { "index": 19, "weight": 1 }),
    ("zombie",           { "index": 20, "weight": 1 }),
    ("skeleton",         { "index": 21, "weight": 1 }),
    ('arrow-left',       { "index": 22, "weight": 1 }),
    ('arrow-right',      { "index": 23, "weight": 1 }),
    ('arrow-up',         { "index": 24, "weight": 1 }),
    ('arrow-down',       { "index": 25, "weight": 1 }),
    ('plant-ripe',       { "index": 26, "weight": 1 }),
    ('plant',            { "index": 27, "weight": 1 }),
    ("fence",            { "index": 28, "weight": 1 }),
])
index_first_object = 13
object_keys = list(objects.keys()) # for indexed access
object_weights = [objects[x]['weight'] for x in objects]
