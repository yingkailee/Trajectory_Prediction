1. Download [SUMO!](https://www.eclipse.org/sumo/)
2. A network file can be created through the installed netedit application or OpenStreetMap.
3. For OpenStreetMap, export an area to create a .osm file which can be converted to a network file through
```shell
netconvert --osm-files OSM_FILENAME.osm -o NET_FILENAME.net.xml
```
4. A routes file is necessary to run a simulation. This can be manually done through editing an xml file, or use SUMO's randomTrips.py tool which is located in SUMO directory.
```shell
(python or py) PATH/TO/randomTrips.py -n NET_OUTPUT_FILENAME.net.xml -r ROUTES_FILENAME.rou.xml -e 100 -l 
```
5. Create a SIMULATION_NAME.sumocfg file.
```xml
<configuration>
 <input>
  <net-file value="NET_FILENAME.net.xml"/>
  <route-files value="ROUTES_FILENAME.rou.xml"/>
 </input>
 <time>
  <begin value="0"/>
  <end value="1000"/>
  <step-length value="0.1"/>
 </time>
</configuration>
```
6. The simulation can be then run from terminal using the ```sumo-gui``` command or by opening the application and selecting the simulation file.
