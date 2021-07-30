from argparse import ArgumentParser
import copy
from easydict import EasyDict as edict
import logging
import logging.config
import numpy as np
import os
import sys
import torch

from lib import cfg_utils, fdutil, io_control_file, rasterutils, utils, validate_cfg_inference
from lib.formatter import RawFormatter
from lib.UNet import UNet
from lib.evaluation import evaluate_performance, predict_linear_blend, get_statistics, print_statistics


RESIDUAL_THRESHOLD = None


parser = ArgumentParser(description='ResDepth:\nA Deep Prior For 3D Reconstruction From High-resolution Satellite '
                                    'Images (Testing)',
                        formatter_class=RawFormatter)

parser.add_argument('config_file', type=str, help='JSON configuration file')


def main():
    # Parse the command line arguments
    args = parser.parse_args()
    cfg_file = args.config_file

    prog_name = 'Running ResDepth: Prediction'
    print('\n{}\n{}\n'.format(prog_name, '=' * len(prog_name)))

    if not fdutil.file_exists(cfg_file):
        print('ERROR: Cannot find the configuration file: {}'.format(cfg_file))
        sys.exit(1)

    # Set up Logger
    logger = utils.setup_logger('root_logger', level=logging.INFO, log_to_console=True, log_file=None)

    # Verify the configuration file
    print(f'Validate the configuration file:\t{cfg_file}\n\n')
    eval_cfg = validate_cfg_inference.validate_and_update_cfg_file(cfg_file, logger)

    if eval_cfg.status is False:
        sys.exit(1)
    else:
        cfg = copy.deepcopy(eval_cfg.cfg)
        cfg_orig = cfg_utils.read_json(cfg_file)
        del eval_cfg

    # Add a file handler to the logger
    utils.add_file_logger(logger, log_file=os.path.join(cfg.output.directory, 'run.log'))

    # -------------------------- Data allocation, normalization parameters -------------------------- #
    logger.info('Perform data allocation...')
    utils.allocate_area(cfg)

    logger.info('\nData initialization\n-------------------\n')
    if cfg.model.input_channels != 'geom':
        logger.info('Read image pairs...\n')
        if utils.read_image_pairs(cfg, logger) is False:
            sys.exit(1)

    logger.info('Read normalization parameters...')
    params_dsm = edict(io_control_file.read_normalization_params_from_file(cfg.model.normalization_geom))

    if cfg.model.input_channels != 'geom':
        params_images = edict(io_control_file.read_normalization_params_from_file(cfg.model.normalization_image))

    # Extract the definition of the test dataset(s)
    cfg_data = utils.prepare_dataset_configuration(cfg, phase='test')

    # Save normalization parameters
    for i, dataset in enumerate(cfg_data):
        dataset.dsm_mean = None
        dataset.dsm_std = params_dsm['std']
        if cfg.model.input_channels != 'geom':
            dataset.images_mean = params_images['mean']
            dataset.images_std = params_images['std']

    # ----------------------------------------- Load model ----------------------------------------- #
    logger.info('\n\nDefine model\n------------\n')
    logger.info('Initialize model...')

    if cfg.model.name == 'UNet':
        model = UNet(**cfg.model.settings)
    else:
        logger.error('Unknown model architecture.\n')
        sys.exit(1)

    # Load weights
    logger.info(f"Load model weights: {cfg.model.weights}")
    model.load_state_dict(torch.load(cfg.model.weights)['model_state_dict'])
    model.eval()

    # ----------------------------------------- Apply model ----------------------------------------- #
    logger.info('\n\nInference\n---------\n')
    logger_index = -1

    for index, dataset in enumerate(cfg_data):
        utils.print_dataset_name_to_console(dataset, index, logger)

        area_to_predict = f"_{dataset.area_type}_area" if 'area_type' in dataset else ''
        name = dataset.name if 'name' in dataset else 'dataset_' + str(index)

        # Create an output directory per test dataset
        output_directory_parent = os.path.join(cfg.output.directory, name)
        fdutil.make_dir(output_directory_parent)

        # Write the original user input configuration to file
        config_file = os.path.join(output_directory_parent, 'config.json.orig')
        cfg_utils.write_json(cfg_orig, config_file)

        # Write the final configuration to file
        config_file = os.path.join(output_directory_parent, 'config.json')
        cfg_utils.write_json(cfg, config_file)

        if cfg.model.input_channels != 'geom':
            image_pairs = dataset.image_pairs
        else:
            # Dummy variable such that the for-loop below is accessible for evaluating ResDepth-0
            image_pairs = [None]

        # Extract the name of the initial DSM raster
        basename = fdutil.filename_wo_ext(dataset.raster_in)

        # Initialize the residual errors over all stereo pairs
        list_all_residuals = []
        list_all_residuals_building = []
        list_all_residuals_terrain = []
        list_all_residuals_terrain_nowater = []
        list_all_residuals_terrain_nowater_noforest = []

        for p, image_pair in enumerate(image_pairs):
            if cfg.model.input_channels != 'geom':
                # Create a subdirectory for each evaluated image (pair)
                if len(image_pair) == 1:
                    foldername = 'Image'
                elif len(image_pair) == 2:
                    foldername = 'Stereopair'
                else:
                    foldername = 'Imagepair'

                list_image_names = []
                for image_index in image_pair:
                    foldername += f'_{image_index}'
                    list_image_names.append(fdutil.filename(dataset.image_list[image_index]))

                output_directory = os.path.join(output_directory_parent, foldername)
                fdutil.make_dir(output_directory)

                # Print image (pair) being evaluated
                if len(list_image_names) == 1:
                    logger.info(f'\n\nInference using the following image:')
                    logger.info(f'Image {image_pair[0]}:\t{list_image_names[0]}\n')

                elif len(list_image_names) == 2:
                    logger.info(f'\nInference using the following stereo pair:   {image_pair}')
                    logger.info(f'Image {image_pair[0]}:\t{list_image_names[0]}')
                    logger.info(f'Image {image_pair[1]}:\t{list_image_names[1]}\n')
                else:
                    logger.info(f'\nInference using the following images:   {image_pair}')
                    for k, image_name in enumerate(list_image_names):
                        logger.info(f'Image {image_pair[k]}:\t{image_name}')
                    logger.info('\n')

                dataset.image_pairs = [image_pair]
                dataloader = utils.get_dataloader([dataset], sampling_strategy='test',
                                                  transform_dsm=True, transform_orthos=True,
                                                  use_all_stereo_pairs=False, permute_images_within_pair=False,
                                                  input_channels=cfg.model.input_channels,
                                                  tile_size=cfg.general.tile_size, augment=False, batch_size=1,
                                                  shuffle=False, workers=cfg.general.workers, pin_memory=True)

            else:
                logger.info(f'Inference without image guidance.\n')
                output_directory = output_directory_parent

                dataloader = utils.get_dataloader([dataset], sampling_strategy='test',
                                                  transform_dsm=True, transform_orthos=False,
                                                  use_all_stereo_pairs=False, permute_images_within_pair=False,
                                                  input_channels=cfg.model.input_channels,
                                                  tile_size=cfg.general.tile_size, augment=False, batch_size=1,
                                                  shuffle=False, workers=cfg.general.workers, pin_memory=True)

            logger.info('Predict...')
            prediction = predict_linear_blend(dataloader, model)

            if 'raster_gt' in dataset:
                logger.info('Evaluate...')

                # Evaluate performance and write statistics to file
                filename = f'{basename}_prediction{area_to_predict}_statistics.txt'
                file_stats = os.path.join(output_directory, filename)

                logger_index += 1
                logger_stats = utils.setup_logger(f'stats_logger{logger_index}', level=logging.INFO,
                                                  log_to_console=False, log_file=file_stats)
                logger_stats.info(f"Model name:\t{cfg.model.name}")
                logger_stats.info(f"Model weights:\t{cfg.model.weights}\n\n\n")
                utils.add_console_logger(logger_stats)

                residuals = evaluate_performance(prediction, dataloader.dataset.dsm_input_gdal,
                                                 dataloader.dataset.dsm_target_gdal, logger,
                                                 dataset.area_defn, dataset.mask_ground_truth, dataset.mask_building,
                                                 dataset.mask_water, dataset.mask_forest,
                                                 logger_stats, RESIDUAL_THRESHOLD)

                logger.info('\n\nSave prediction...')

                # Number of regions specified in area_defn
                num_regions = len(dataloader.dataset.area_defn.x_extent)

                # Export the refined DSM and its residual error map as GeoTiffs
                for i in range(num_regions):
                    x = dataloader.dataset.area_defn.x_extent[i]
                    y = dataloader.dataset.area_defn.y_extent[i]

                    if num_regions == 1:
                        name1 = f'{basename}_prediction{area_to_predict}.tif'
                        name2 = f'{basename}_residuals{area_to_predict}.tif'
                    else:
                        name1 = f'{basename}_prediction{area_to_predict}_{i}.tif'
                        name2 = f'{basename}_residuals{area_to_predict}_{i}.tif'

                    file_prediction = os.path.join(output_directory, name1)
                    file_residuals = os.path.join(output_directory, name2)

                    prediction_i = prediction[y[0]:y[1]+1, x[0]:x[1]+1]
                    residuals_i = residuals.all[y[0]:y[1]+1, x[0]:x[1]+1]

                    # Save the residual errors for later (to compute statistics averaged over all predictions)
                    list_all_residuals.append(residuals_i.compressed())

                    # Replace masked pixels with the nodata value -9999
                    residuals_i[residuals_i.mask] = -9999

                    logger.info('Write file: {}'.format(file_prediction))
                    rasterutils.export_data_as_raster(dataloader.dataset.dsm_input_gdal, file_prediction,
                                                      prediction_i, x[0], y[0], nodata=-9999)

                    logger.info('Write file: {}'.format(file_residuals))
                    rasterutils.export_data_as_raster(dataloader.dataset.dsm_input_gdal, file_residuals,
                                                      residuals_i, x[0], y[0], nodata=-9999)

                    if 'building' in residuals:
                        list_all_residuals_building.append(residuals.building[y[0]:y[1]+1, x[0]:x[1]+1].compressed())
                        list_all_residuals_terrain.append(residuals.terrain[y[0]:y[1]+1, x[0]:x[1]+1].compressed())

                    if 'terrain_nowater' in residuals:
                        list_all_residuals_terrain_nowater.append(residuals.terrain_nowater[y[0]:y[1]+1,
                                                                  x[0]:x[1]+1].compressed())

                    if 'terrain_nowater_noforest' in residuals:
                        list_all_residuals_terrain_nowater_noforest.append(residuals.terrain_nowater_noforest[y[0]:y[1]+1,
                                                                           x[0]:x[1]+1].compressed())

                logger.info('Write file: {}\n\n'.format(file_stats))

            else:
                logger.info('\n\nSave prediction...')

                # Number of regions specified in area_defn
                num_regions = len(dataloader.dataset.area_defn.x_extent)

                # Export the refined DSM
                for i in range(num_regions):
                    x = dataloader.dataset.area_defn.x_extent[i]
                    y = dataloader.dataset.area_defn.y_extent[i]

                    if num_regions == 1:
                        name = f'{basename}_prediction{area_to_predict}.tif'
                    else:
                        name = f'{basename}_prediction{area_to_predict}_{i}.tif'

                    file_prediction = os.path.join(output_directory, name)
                    prediction_i = prediction[y[0]:y[1] + 1, x[0]:x[1] + 1]

                    # Replace masked pixels with the nodata value -9999
                    prediction_i[prediction_i == dataloader.dataset.nodata] = -9999

                    logger.info('Write file: {}\n\n'.format(file_prediction))
                    rasterutils.export_data_as_raster(dataloader.dataset.dsm_input_gdal, file_prediction,
                                                      prediction_i, x[0], y[0], nodata=-9999)

        # Compute statistics over all predictions
        if len(image_pairs) > 1 and 'raster_gt' in dataset:
            logger.info('\nCompute residual errors averaged over all predictions...')

            # Concatenate residual errors over all predictions
            all_residuals = np.ma.array(list_all_residuals).flatten()
            del list_all_residuals
            stats = get_statistics(all_residuals, RESIDUAL_THRESHOLD)

            if dataset.mask_building:
                all_residuals_building = np.ma.array(list_all_residuals_building).flatten()
                all_residuals_terrain = np.ma.array(list_all_residuals_terrain).flatten()
                del list_all_residuals_building
                del list_all_residuals_terrain
                stats_building = get_statistics(all_residuals_building, RESIDUAL_THRESHOLD)
                stats_terrain = get_statistics(all_residuals_terrain, RESIDUAL_THRESHOLD)

                if dataset.mask_water:
                    all_residuals_terrain_nowater = np.ma.array(list_all_residuals_terrain_nowater).flatten()
                    del list_all_residuals_terrain_nowater
                    stats_terrain_nowater = get_statistics(all_residuals_terrain_nowater, RESIDUAL_THRESHOLD)

                if dataset.mask_forest:
                    all_residuals_terrain_nowater_noforest = np.ma.array(list_all_residuals_terrain_nowater_noforest).flatten()
                    del list_all_residuals_terrain_nowater_noforest
                    stats_terrain_nowater_noforest = get_statistics(all_residuals_terrain_nowater_noforest,
                                                                    RESIDUAL_THRESHOLD)

            filename = f'{basename}_prediction{area_to_predict}_performance_statistics_mean_over_all_stereopairs.txt'
            outfile = os.path.join(output_directory_parent, filename)

            logger_stats_overall = utils.setup_logger('stats_logger_overall', level=logging.INFO, log_to_console=False,
                                                      log_file=outfile)
            logger_stats_overall.info(f"Model name:\t{cfg.model.name}")
            logger_stats_overall.info(f"Model weights:\t{cfg.model.weights}\n\n\n")
            utils.add_console_logger(logger_stats_overall)

            # Write statistics
            logger_stats_overall.info('\nPerformance Evaluation: Statistics over all predictions\n'
                                      '-------------------------------------------------------\n')

            if RESIDUAL_THRESHOLD:
                logger_stats_overall.info('Truncation threshold:\t\t\t{:.2f} m\n'.format(RESIDUAL_THRESHOLD))

            logger_stats_overall.info('\nSTATISTICS, OVERALL: REFINED DSM\n--------------------------------\n')
            print_statistics(stats, logger_stats_overall)

            if dataset.mask_building:
                logger_stats_overall.info('\nSTATISTICS, BUILDING PIXELS: REFINED DSM\n'
                                          '----------------------------------------\n')
                print_statistics(stats_building, logger_stats_overall)

                logger_stats_overall.info('\nSTATISTICS, TERRAIN PIXELS: REFINED DSM\n'
                                          '---------------------------------------\n')
                print_statistics(stats_terrain, logger_stats_overall)

                if dataset.mask_water:
                    logger_stats_overall.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT WATER: REFINED DSM\n'
                                              '-----------------------------------------------------\n')
                    print_statistics(stats_terrain_nowater, logger_stats_overall)

                    if dataset.mask_forest:
                        logger_stats_overall.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT WATER/FOREST: REFINED DSM\n'
                                                  '------------------------------------------------------------\n')
                        print_statistics(stats_terrain_nowater_noforest, logger_stats_overall)

                elif dataset.mask_forest:
                    logger_stats_overall.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT FOREST: REFINED DSM\n'
                                              '------------------------------------------------------\n')
                    print_statistics(stats_terrain_nowater_noforest, logger_stats_overall)

    logger.info('\nDone!')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
    else:
        main()
