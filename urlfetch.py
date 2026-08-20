"""Fetching over HTTP, and refusing everything else.

urllib's default opener answers `file:`, `ftp:` and `data:` as readily as
it answers `https:`. Every URL this program opens arrives from somewhere
outside it -- a setting, a release feed, a ComfyUI address, a redirect --
and none of those should be able to name a path on the disk.

The opener here is built from the HTTP handlers and no others, so those
schemes are not merely rejected, they are unreachable: with no handler
registered for a scheme, `UnknownHandler` raises `URLError` before
anything is opened. Redirects are confined to http/https/ftp by
`HTTPRedirectHandler` already, and ftp has no handler here either.
"""

from __future__ import annotations

import http.client
import urllib.request

_HANDLERS = [
    urllib.request.ProxyHandler,
    urllib.request.UnknownHandler,
    urllib.request.HTTPHandler,
    urllib.request.HTTPDefaultErrorHandler,
    urllib.request.HTTPRedirectHandler,
    urllib.request.HTTPErrorProcessor,
]
if hasattr(http.client, "HTTPSConnection"):
    _HANDLERS.append(urllib.request.HTTPSHandler)


def _build_opener():
    """An OpenerDirector carrying only the handlers above.

    Built by hand rather than with `urllib.request.build_opener`, which
    always adds FileHandler, FTPHandler and DataHandler unless a subclass
    of each is passed in to displace them.
    """
    opener = urllib.request.OpenerDirector()
    for handler in _HANDLERS:
        opener.add_handler(handler())
    return opener


_OPENER = _build_opener()


class HttpRequest(urllib.request.Request):
    """A request this program is willing to send.

    Refuses anything but http(s) when it is built, which is earlier and
    plainer than the opener refusing it: the ComfyUI address is typed into
    a settings box, so `file:///` is a thing somebody can put there, and
    the answer should name the URL rather than surface as a scheme the
    opener happens to have no handler for.
    """

    def __init__(self, url, **kwargs):
        if not str(url).lower().startswith(("http://", "https://")):
            raise ValueError(f"refusing a URL that is not http or https: {str(url)[:80]!r}")
        super().__init__(url, **kwargs)


def open_url(url, timeout=None):
    """Open `url` over HTTP(S); any other scheme raises `URLError`.

    `url` is a string or a request object. The return value is the same
    response `urllib.request.urlopen` gives, so callers read it and close
    it the same way.
    """
    if timeout is None:
        return _OPENER.open(url)
    return _OPENER.open(url, timeout=timeout)
