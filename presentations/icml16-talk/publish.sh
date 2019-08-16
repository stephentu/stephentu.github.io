#!/bin/bash
rsync -avz --rsync-path=/usr/sww/bin/rsync . sltu@login.eecs.berkeley.edu:~sltu/public_html/wip/icml16-talk
