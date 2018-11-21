all: build

reqs:
	@sudo pip3 install -r requirements.txt 

build: reqs
	@python3 setup.py build

install: build
	@python3 setup.py install

clean:
	@rm -rf build
