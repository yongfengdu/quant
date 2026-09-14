#!/bin/bash
export HTTPS_PROXY=http://proxy-dmz.intel.com:912
export HTTP_PROXY=http://proxy-dmz.intel.com:912
exec python3 -u /root/quant-autoresearch/update_data.py --force >> /tmp/update_data_v3.log 2>&1
