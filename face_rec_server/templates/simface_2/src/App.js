(function($){
    var VERSION = '1.0 DEV';
    window.GET_PROFILES_URL = 'https://simface.net/get_profiles';


    $(document).ready(function() {
        /*
            todo:
                - rewrite to AMD?
        */
        var appView = new AppView({
            model  : new AppModel(),
            version: VERSION,
            el     : 'body'
        });

        appView.render();
    });

})(jQuery);