# Chimp

Code and reproducible tests for comparing Chimp and Gorilla lossless streaming compression algorithms. We provide the means to reproduce our tests for datasets that we use in our work using just a few commands. The repository is forked from https://github.com/influxdata/influxdb and contains all code to build InfluxDB.

## Vagrant (optional)

We include a Vagrantfile to provide an easy setup of the experimental environment. You should have vagrant and virtual box installed. Then simply execute:

```
$ vagrant up  # to create the vm
$ vagrant ssh # to log in the vm
$ cd /vagrant # to switch to the directory that contains the code and proceed to building and testing
````

This will create a virtual machine with all dependencies installed and switch to the directory that contains the code.

## Building

If you chose to omit the previous step please follow the "building from source" guidelines provided in the following link:

https://github.com/influxdata/influxdb/blob/master/CONTRIBUTING.md

Make sure you have go and rustc installed. The version of rustc should be < 1.54.0

```
$ go version
go version go1.13.8 linux/amd64
$ rustc --version
rustc 1.53.0 (53cb7b09b 2021-06-17)
```

While in the root directory (/vagrant inside the virtual machine) build influxdb:

```
$ make
```

## Testing

Testing is as simple as checking out the appropriate branch and executing a single command (you should be in /vagrant directory if you are using the vm):


### Chimp

```
$ git checkout chimp
Switched to branch 'chimp'
$ go test  -test.timeout 0 -run TestCompress_Basel -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 
=== RUN   TestCompress_Basel_Temp
Bits per value: 54.1238064516129
Compression time per block 163441.75806451612
Decoding time per block 35765.22580645161
--- PASS: TestCompress_Basel_Temp (0.15s)
=== RUN   TestCompress_Basel_Wind_Speed
Bits per value: 56.115870967741934
Compression time per block 158.82009677419356
Decoding time per block 34.12603225806452
--- PASS: TestCompress_Basel_Wind_Speed (0.16s)
PASS
ok  	github.com/influxdata/influxdb/v2/tsdb/engine/tsm1	0.325s
```

### Chimp128

```
$ git checkout chimp128
Switched to branch 'chimp128'
$ go test  -test.timeout 0 -run TestCompress_Basel -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 
=== RUN   TestCompress_Basel_Temp
Bits per value: 29.35258064516129
.
.
.
```

### Gorilla
```
$ git checkout gorilla
Switched to branch 'gorilla'
$ go test  -test.timeout 0 -run TestCompress_Basel -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 
=== RUN   TestCompress_Basel_Temp
Bits per value: 60.208193548387094
.
.
.
```

### Gorilla128

```
$ git checkout gorilla128 
Switched to branch 'gorilla128'
$ go test  -test.timeout 0 -run TestCompress_Basel -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1 
=== RUN   TestCompress_Basel_Temp
Bits per value: 65.22103225806451
.
.
.
```

## Clean up

```
$ git checkout master
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
$ vagrant halt; vagrant destroy -f
==> default: Attempting graceful shutdown of VM...
==> default: Destroying VM and associated drives...
```

## Who do I talk to?

* Panagiotis Liakos: http://cgi.di.uoa.gr/~p.liakos/
* Katia Papakonstantinopoulou: www.aueb.gr/users/katia
* Yannis Kotidis http://pages.cs.aueb.gr/~kotidis/
