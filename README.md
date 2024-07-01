1. Start Server: `CarlaUE4.exe -carla-server`
2. Run client: `py manual_driving.py`

TODO:
1. Compute curvature at several displacements in front of AV.
2. Compute immediate heading delta.
2.5 I would have to produce distance to closest traffic light since traffic manager is slowing down when traversing an intersection.
3. Refactor my code out.
4. Save features as a data set.
5. Start looking into the model.