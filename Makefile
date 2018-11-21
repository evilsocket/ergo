all: build

reqs:
	@sudo pip3 install -r requirements.txt 

build: reqs
	@python3 setup.py build

install: build
	@python3 setup.py install

train:
	@/usr/local/bin/ergo train test-is-malware --dataset test-is-malware/ergo.is_malware.data

train_fast:
	@/usr/local/bin/ergo train test-is-malware

clean:
	@rm -rf test-is-malware/*.csv
	@rm -rf build
