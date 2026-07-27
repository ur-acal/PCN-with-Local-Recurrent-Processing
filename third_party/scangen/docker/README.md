# README.md

Make a container and a volume.
```bash
docker compose build
```
Copy all files in the data directory to that volume.
```bash
make volume
```
Start the container with
 - Your user and group IDs.
 - A bind mount for the source code in the current directory.
 - A volume mount for the data.
```bash
cd scangen
docker compose -f docker/docker-compose.yml run --rm scangen /bin/bash
```
