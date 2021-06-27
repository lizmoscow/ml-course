var UploadView = Backbone.View.extend({
    template: _.template($('#upload-template').html()),
    inactiveOpacity: '0.25',

    events: {
        'mouseenter'   : 'active',
        'mouseleave'   : 'inactive',
        'click'        : 'open',
        'change #file' : 'input',

        'drag'         : 'prevent',
        'dragstart'    : 'prevent',
        'dragend'      : 'prevent',
        'dragenter'    : 'prevent',
        'dragleave'    : 'prevent',

        'dragover'     : 'dragover',

        'dragenter'    : 'active',

        'dragleave'    : 'inactive',
        'dragend'      : 'inactive',
        'drop'         : 'drop'
    },

    initialize: function(argc) {
        Object.assign(this, argc);
    },

    dragover: function(e) {
        this.prevent(e);
        this.active();
    },

    drop: function(e) {
        this.prevent(e);

        this.inactive();

        this.loadFile(e.originalEvent.dataTransfer.files[0]);
    },

    prevent: function(e) {
        e.preventDefault();
        e.stopPropagation();
    },

    active: function() {
        this.$('.upload-border').css('opacity', '1');
    },

    inactive: function() {
        this.$('.upload-border').css('opacity', this.inactiveOpacity);
    },

    open: function(e) {
        e.preventDefault();
        e.stopPropagation();

        this.$('#file').trigger('click.input');
    },

    upload: function() {
        var $this = this;

        //var file_data = $('#file').prop('files')[0];
        var data = new FormData();
        data.append('id1', this.file);

        $.ajax({
            url: window.GET_PROFILES_URL,
            type: 'POST',
            data: data,
            cache: false,
            dataType: 'json',
            processData: false,
            contentType: false,
            success: function( respond, textStatus, jqXHR ){
                if( typeof respond.error === 'undefined' ){
                    console.log(respond);
                    $this.showResult(respond.result);
                }
                else{
                    console.log('error: ' + respond.error );
                }
            },
            error: function( jqXHR, textStatus, errorThrown ){
                console.log('error: ' + textStatus );
            }
        });
 
        
    },

    _base64ToArrayBuffer: function (base64) {
        var binary_string = window.atob(base64);
        var len = binary_string.length;
        var bytes = new Uint8Array(len);
        for (var i = 0; i < len; i++) {
            bytes[i] = binary_string.charCodeAt(i);
        }
        return bytes.buffer;
    },

    showResult: function(result) {
        Backbone.trigger('profileList.setProfiles', result);
        Backbone.trigger('flipper.upload&profiles');
        Backbone.trigger('refresh.animationShow');
    },

    loadFile: function(file) {
        var $this = this;

        this.file = file;

        this.model.load(file, function(e) {
            var base64 = e.target.result;

            $this.model.push(e.target.result);

            Backbone.trigger('photo.setPhoto', base64);
            Backbone.trigger('flipper.logo&photo');

            $this.upload();
        });

        this.undelegateEvents();
        this.active();
        this.$('#upload-spinner').removeClass('is-hidden');
        this.$('#upload-icon').addClass('is-hidden');
    },

    input: function(e) {
        var files = e.originalEvent.target.files;

        if (!files)
            return;

        this.loadFile(files[0]);
    },

    render: function() {
        this.$el.html(
            this.template
        );

        this.inactive();

        return this;
    }
});

_.extend(UploadView.prototype, AnimationMixin);