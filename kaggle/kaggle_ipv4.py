"""Chạy Kaggle CLI nhưng chỉ dùng IPv4.

Lý do (13/11/2026): mạng của máy không đi được IPv6 tới Google (kết nối IPv6 tới
storage.googleapis.com hết giờ, IPv4 xong trong 0,06 s). Lúc tải file lên, CLI dùng
urllib3, thư viện này thử IPv6 trước và treo; curl thì tự chuyển sang IPv4 nên không lỗi.

Cách dùng: <python của Kaggle CLI> kaggle/kaggle_ipv4.py <các tham số như lệnh kaggle>
(push.sh tự gọi theo cách này).
"""
import sys

import urllib3.util.connection as _conn

_conn.HAS_IPV6 = False      # allowed_gai_family() sẽ trả về AF_INET

from kaggle.cli import main  # noqa: E402

sys.argv[0] = "kaggle"
sys.exit(main())
