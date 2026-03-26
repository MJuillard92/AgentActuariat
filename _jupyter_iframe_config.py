c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors * 'self'",
        'X-Frame-Options': '',
    }
}
c.ServerApp.allow_origin = '*'
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True
c.ServerApp.open_browser = False
