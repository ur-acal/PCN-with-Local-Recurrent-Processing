.PHONY: container volume clean


build:
	docker compose build

volume:
	docker run --rm -v docker_7528s01_photos:/data -v $(realpath ../../data):/source alpine cp -a /source/. /data/

clean:
	-docker image rm 7528s01_scangen
	-docker volume rm 7528s01_photos
