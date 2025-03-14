import requests
import threading
import concurrent.futures
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr  
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats 
import os
import cftime
from concurrent.futures import ThreadPoolExecutor
from IPython.display import Markdown
import inspect
###################################################################################################################################
##                                           Meta Data Helper Functions
###################################################################################################################################
def extract_varientID(string):  # also Known as extract_8_chars
    start_index = string.find('_r') 
    end_index = string.find('_', start_index+1)
    return string[start_index+1:end_index]

def extract_monType(string):  # also Known as extract_8_chars
    start_index = string.find('_') 
    end_index = string.find('_', start_index+1)
    return string[start_index+1:end_index]

def extract_SimNums(string):
    try:
        start_index = string.find('_r')
        end_index = string.find('i', start_index)
        r = int(string[start_index + 2:end_index])

        start_index = string.find('i', end_index)
        end_index = string.find('p', start_index)
        i = int(string[start_index + 1:end_index])

        start_index = string.find('p', end_index)
        end_index = string.find('f', start_index)
        p = int(string[start_index + 1:end_index])

        start_index = string.find('f', end_index)
        end_index = string.find('_', start_index)
        f = int(string[start_index + 1:end_index])
        
        return r, i, p, f
    except Exception as e:
        # Provide feedback for debugging
        print(f"Error processing string '{string}': {e}")
        return None, None, None, None
   
def getPeriod(filename):
    if 'abrupt-4xCO2' in filename: return 'abrupt-4xCO2'
    elif 'piControl' in filename: return 'piControl'
    elif 'historical' in filename: return 'historical'
    elif 'ssp245' in filename: return 'ssp245'
    
def whichVar(filename):
    start_index = filename.find('_')
    return filename[:start_index]

def whichTimestep(filename):
    if 'day' in filename: return 'day'
    elif 'mon' in filename: return 'mon'
    else : return np.nan

def whichGrid(filename):
    varID = extract_varientID(filename)
    start_index = filename.find(varID) +len(varID)
    end_index = filename.find('_', start_index+1)
    return filename[start_index+1:end_index]

def extractDates(string):
    ncFind = string.find('.nc') 
    ncBack = string.rfind('-', 0, ncFind)
    dback = (ncFind-ncBack)-1
    stop = string[ncBack+1:ncFind]
    start = string[ncBack-dback:ncBack]
    try:
        return extractYear(start), extractYear(stop)
    except:
        print(string)
def extractYear(Date):
    return int(Date[:4])
def getExperiment(filename):
    if 'abrupt-4xCO2' in filename: return '4x'
    elif 'piControl' in filename: return '4x'
    elif 'historical' in filename: return 'SSP245'
    elif 'ssp245' in filename: return 'SSP245'

###################################################################################################################################
##                                           Parallel Processor
###################################################################################################################################
def parallel_execution(func, inputs, processes=None):
    """my generic parallel processing function"""
    from multiprocessing import Pool, cpu_count

    # Set the number of processes to use (default: number of CPU cores)
    if processes is None:
        processes = cpu_count()

    # Create a pool of worker processes
    with Pool(processes=processes) as pool:
        # Map the function to inputs and distribute across processors
        results = pool.map(func, inputs)

    return results

###################################################################################################################################
##                                           File Search Infrastructure
###################################################################################################################################
def fileContextSearch(inputs):
    '''Helper function for seachFPs: puts together a dictionary with search results'''
    res, model = inputs
    hit = res.file_context().search()
    files = [
        {
            'model': model,
            'filename': f.filename, 
            'download_url': f.download_url, 
            'opendap_url': f.opendap_url
        } 
        for f in hit
        ]
    return files

def searchFPs(conn, model, variables, tStep='day', period='SSP245'):
    '''given model and variables looks for CMIP6 models filling criteria
    returns as flattened df - consider saving along the way'''
    query = conn.new_context(
            project="CMIP6",     
            experiment_id=period,
            source_id=model,
            frequency=tStep,
            variable_id=variables
        )
    ## Looking for Data
    results = query.search()
    inputs = list(zip(results, np.repeat(model, len(results))))
    
    files = parallel_execution(fileContextSearch, inputs)
    
    flat_files = [file for sublist in files for file in sublist]

    return pd.DataFrame(flat_files)

def addMetaData(df):
    '''Adds some additional columns for df'''
    df['Varient'] = df.filename.apply(extract_varientID)
    df['period'] = df.filename.apply(getPeriod)
    df['Var'] = df.filename.apply(whichVar)
    df['grid'] = df.filename.apply(whichGrid)
    df['timeStep'] = df.filename.apply(whichTimestep)
    df[['start', 'stop']] = df['filename'].apply(extractDates).apply(pd.Series)
    df[['r', 'i', 'p', 'f']] = pd.DataFrame(
                df['filename'].apply(extract_SimNums).tolist(),
                index=df.index)
    return df

def pickVarientandFilter(df):
    '''This function takes a df with filenames and meta data (specifically varient ids) and finds shared varients for each model and period
    Minimizing varient number to keep toward r1f1p1i1 although deviations in the f and p components are likely'''
    shared_combinations = {}
    
    for (period, model), group in df.groupby(['period', 'model']):
        combinations_per_variable = (
            group.groupby('Var')[['r', 'f', 'p', 'i']]
            .apply(lambda var_group: set(tuple(x) for x in var_group.to_numpy()))
            .tolist()
        )
        if combinations_per_variable:
            shared_combinations_for_group = set.intersection(*combinations_per_variable)
        else:
            shared_combinations_for_group = set()
        shared_combinations[(period, model)] = shared_combinations_for_group
    
    filtered_dfs = []
        
    for (period, model), combinations in shared_combinations.items():
        if combinations:  
            first_combination = sorted(list(combinations))[0]
            
            filtered_df = df[
                (df['period'] == period) & 
                (df['model'] == model) & 
                (df[['r', 'f', 'p', 'i']].apply(tuple, axis=1) == first_combination)
            ]
            filtered_dfs.append(filtered_df)
    
    filtered_dfs = pd.concat(filtered_dfs, ignore_index=True)

    return filtered_dfs

###################################################################################################################################
##                                           Download Infrastructure
###################################################################################################################################
def download_concurrently(args):
    """This one is a mystery, takes args which are a tuple of paths (different download paths) for file i
    it starts a new thread for each seperate path with a shared signal (Stop event) It tries to download each file simultaniously but bc each file tries to download with the same name there are temporary file names that are given. The download and check function does the actual downloading  and checking to see if the file is larger than 1kb

    When it succesfully downloads a file larger than a kb it sends the stopevent signal to end all threads it created
    it then clears all but the sucessfully downloaded file and then renames the good file to what it was meant to be named in the first place 
    """

    paths, i = args[0], args[1]
    stop_event = threading.Event()  # Each call has its own stop signal

    downloadPaths = paths[paths.filename == i].reset_index(drop=True)
    final_filename = f'TempData/{i}'

    # Create temporary filenames
    temp_filenames = [f'TempData/{i}_temp_{idx}' for idx in range(len(downloadPaths))]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(download_and_check, row['download_url'], temp_filename, final_filename, stop_event): (row['download_url'], temp_filename)
            for (_, row), temp_filename in zip(downloadPaths.iterrows(), temp_filenames)
        }

        try:
            for future in concurrent.futures.as_completed(futures):
                url, temp_filename = futures[future]
                try:
                    success, valid_temp_filename = future.result()
                    
                    if success:
                        
                        stop_event.set()  # Signal all threads in this batch to stop
                        
                        # Cancel remaining futures (prevents unstarted tasks)
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        
                        executor.shutdown(wait=False, cancel_futures=True)  # Forcefully stop executor
                        
                        # Remove other temp files
                        for other_temp_filename in temp_filenames:
                            if other_temp_filename != valid_temp_filename and os.path.exists(other_temp_filename):
                                os.remove(other_temp_filename)

                        # Rename the successful download to final filename
                        os.rename(valid_temp_filename, final_filename)
                        return
                
                except concurrent.futures.TimeoutError:
                    print(f"Download timed out: {url}")
        
        except KeyboardInterrupt:
            print("\nInterrupted! Stopping downloads...")
            stop_event.set()  # Signal threads in this batch to stop
            executor.shutdown(wait=False, cancel_futures=True)  # Force shutdown
            raise  # Re-raise to terminate execution
        except Exception as e:
            print(e, ' Somewhere in  download_concurrently')
            
def download_and_check(url, temp_filename, final_filename, stop_event):
    """Download file but stop if the event is set."""
    try:
        r = requests.get(url, stream=True, timeout=10)
        r.raise_for_status()
        
        with open(temp_filename, "wb") as f:
            for chunk in r.iter_content(1024):
                if stop_event.is_set():  # Actively check stop condition
                    return False, None
                f.write(chunk)

        return True, temp_filename  # Return success flag and valid temp file
    
    except Exception as e:
        return False, None
        
def preprocess_and_download(models, df, preprocessFunc, maxWorkers=2):
    ''' This is where we get a little more theoretical - Lets assume that we want to download the raw data for each of given variables and periods
    periods - more of a relic of my own work but could be like historical or pmip'''
    tasks = [
        (period, Var, df, model, preprocessFunc)
        for period in df.period.unique()
        for Var in df.Var.unique()
        for model in models   
    ]
    ## The key thing here is the process_period_var function - this will do the heavy lift for downloading and preprocessing (if we need preprocessing)
    with ThreadPoolExecutor(max_workers=maxWorkers) as executor: ### Ramp up max worker if needed
        executor.map(lambda args: process_period_var(*args), tasks)
    


def process_period_var(period, Var, df, model, preprocessFunc):
    """Handles downloading data for a single (period, variable) combination using download concurrently - inputs are split up by file path unique
    """
    paths = df[((df.model == model) & (df.period == period))&(df.Var == Var)].reset_index(drop=True)
    saveName = f'TempData/{model}_{period}_{Var}_processed.nc'

    if not os.path.exists(saveName):  # Check if processed dataset has already been created
        try:
            # Prepare inputs for parallel processing of file downloads
            inputs = [(paths, i) for i in paths.filename.unique()]
            print(f'Downloading data for {saveName}')
            parallel_execution(download_concurrently, inputs, processes = 8)  # Keep using multiprocessing for downloads
            
            # Open Raw Data
            ds = xr.open_mfdataset(sorted([f'TempData/{fn}' for fn in paths.filename.unique()]), combine='nested', concat_dim='time', use_cftime=True)
            ds = ds[Var]
            # Preprocess Here: vvv
            ds = ds.chunk({'time':-1, 'lat':20})
            ds = preprocessFunc(ds, period, Var, df, model)
            # Save Preprossed Data
            write_job = ds.to_netcdf(saveName, compute=False)
            
            with ProgressBar():
                print(f"Writing to {saveName}")
                write_job.compute()
            for k in [f'TempData/{fn}' for fn in paths.filename]:
                if os.path.exists(k): os.remove(k)
            for filename in os.listdir('TempData/'):
                file_path = os.path.join('TempData/', filename)
        
                if ("_temp" in filename) & (os.path.getsize(file_path) < 1024):
                    os.remove(file_path)
        
        except Exception as e:
            print(e)


###################################################################################################################################
##                                           Other Stuff
###################################################################################################################################
def show_function_markdown(func_name):
    code = inspect.getsource(func_name)
    return Markdown(f"```python\n{code}\n```")

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
# size = sizeof_fmt(os.path.getsize(f'TempData/{model}_historical_tas_processed.nc'))