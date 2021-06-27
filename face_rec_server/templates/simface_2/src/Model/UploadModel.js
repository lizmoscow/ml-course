var UploadModel = Backbone.Model.extend({
    defaults: {
        files: []
    },

    load: function(file, callback) {
        var fileReader = new FileReader();
        
        fileReader.onloadend = callback;
    
        fileReader.readAsDataURL(file);
    },

    loadAll: function(fileList, callback) {
        for (var i = 0; i < fileList.length; i++)
            this.load(fileList[i], callback);
    },

    push: function(file) {
        var files = this.get('files');
        files.push(file);
    }

});