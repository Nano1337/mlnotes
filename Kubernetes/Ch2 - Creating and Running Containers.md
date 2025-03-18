
**Basic Definitions**
- container image: binary package that encapsulates all files necessary to run program inside OS container
	- combined with container config file to provide instr on how to set up container env and execute app from entry point
- container config file: includes info about how to set up networking, namespace isolation, resource constraints (cgroups), syscall restrictions
- container registry: where images are stored (either remote or local)
- Overlay system: each layer adds, removes, or modifies files from preceding layer in filesystem. This is the conceptual framework in that Docker images build on top of existing ones
	- each parent reference is a pointer, real-world containers can be part of DAGs
- System container: mimic VMs and run full boot process - not good practice nowadays
- Application container: run single program (perfect lvl of abstraction)

**Building App Images with Docker**
- Dockerfile: used to automate creation of Docker container image
- .dockerignore (like a gitignore): defines set of files that should be ignored when copying files into image, e.g. `node_modules` 
- Process: 
	- specify base image (FROM)
	- specify work directory (WORKDIR)
	- Copy over files, run setup, install dependencies
	- Run default command to start program
- Example: 
  ```bash
FROM node:16

WORKDIR /usr/src/app

COPY package*.json ./ 
RUN npm install 
RUN npm install express

COPY ..

CMD ["npm", "install"]
```

Then run `docker build -t <name>` and run local image `docker run --rm -p 3000:3000 <name>`

**Optimizing Image Sizes**
There are two situations we want to be careful of: 
1. Adding and removing large files in different images. Assuming the BigFile is in the base layer, removing it in later layers would still require BigFile to be transmitted through the network, even if you can no longer access it
	1. this also applies to private keys due to this layer versioning
2. You should keep the code that's most frequently changed in the outermost layer. This bc if it's in a more base layer then

**MultiStage Image Builds**
Putting program compilation (and thus all related dependencies for compilation) can lead to a very heavy image. We need to decouple the build and deployment stages of images. For example: 
```bash 
# Stage 1: Build dependencies and compile program

FROM golang:1.17-alpine AS build

RUN ...
WORKDIR /go/src/.../kuard

COPY ..

ENV .....

RUN build/build.sh

# Stage 2: Copy over compiled program and deploy/use it

FROM alpine

COPY --from=build /go/bin/kuard /kuard 

CMD ["/kuard"]

```

**Running Containers with Docker**
- daemon: background process that runs on system without direct user interaction; usually started at boot time and continue to run
- kubelet: daemon that launches docker containers on each node in k8s
Example: 
```bash
docker run -d --name kuard \
--public 8080:8080 \
gcr.io/kuar-demo/kuward-amd64:blue
```
- `-d` specifies that the process should run as a daemon in the background
- `--name` makes it have a readable name
- `--public` or `-p` allows for port forwarding to connect 8080 on localhost to port 8080 in the container

**Limiting Resource Usage in Docker**
```bash 
--memory 200m \
--memory-swap 1G \
--cpu-shares 1024 \
```
- these three flags limit the number of physical resources that the docker container can use. If the processes inside the docker container use more than these limits then it will be terminated

**Cleanup**
- `docker rmi <tag-name/image-id>` will delete images. Unless you explicitly delete docker images, then it'll live locally forever
- Can delete unused images automatically through `docker system prune` and can even schedule cron job to do this for you: 
  ```bash
crontab -e

0 3 * * * /usr/bin/docker system prune
```
- runs cmd everyday at 3am

**How cron jobs work**
- cron is a daemon process
- uses config files called crontabs to determine what/when to run
  ```bash
* * * * * command-to-be-executed
│ │ │ │ │
│ │ │ │ └─── Day of the week (0-7) (both 0 and 7 represent Sunday)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of the month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
```

