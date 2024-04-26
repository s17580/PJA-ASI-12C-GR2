# Conda Commands

Install Anaconda Link: https://www.continuum.io/downloads

1. Check/List installed packages: <br>
	`Conda list`

2. Upgrade Conda packages:<br>
	`conda upgrade conda
	conda upgrade --all`

3. Install conda package:<br>
	`conda install package_name`

4. Installing multiple packages:<br>
	`conda install numpy scipy pandas`

5. To specify which version of a package you want by adding the version number:<br>
	`conda install numpy=1.10`

6. To uninstall package:<br>
	`conda remove package_name`

7. To update a package:<br>
	`conda update package_name`

8. To update all packages:<br>
	`conda update --all`

9. If you are not familiar with the exact name you are searching for, try searching with: <br>
	`conda search search_term`

10. Create an environment:<br>
	`conda create -n env_name list of packages`<br>

	-n env_name       --> sets name of your environment<br>
	list of packages  --> list of packages you want installed in the
			      environment.<br>
	`example:
	 conda create -n my_env numpy
	 
	 conda create -n py3 python=3
	 
	 conda create -n py python=3.3 `<br>

11. Entering an environment:<br>
	`source activate my_env`

12. Leaving an environment:<br>
	`source deactivate my_env`

13. Save packages of the environment to YAML file:<br>
	`conda env export > environment.yaml`<br>
	conda env export --> writes out all the packages in the environment				including python version.<br>

14. To check all the versions of the packages:<br>
	`conda env export`

15. To create an environment from an environment file:<br>
	` conda env create -f environment.yaml`

16. To list all the environments:<br>
	`conda env list`

17. Removing environments:<br>
	`conda env remove -n env_name`
	
18. Sharing Environments:
	When sharing your code on GitHub, it's good practice to make an environment file and include it in the repository.
	This will make it easier for people to install all the dependencies for your code.
	Include a `pip requirements.txt` file using `pip freeze` (learn more here) for people not using conda. <br>

	`pip freeze > requirements.txt`

	